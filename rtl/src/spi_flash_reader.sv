// SPI NOR flash read controller: sequential byte stream with backpressure.
//
// start + addr + mode begins a read; bytes then come out on out_data/out_valid/out_ready
// for as long as the consumer takes them. When the consumer stalls, SCK is held low with
// CS# still asserted (NOR flash is static, the read resumes where it stopped). stop ends the
// read, drops any bytes not yet taken, and after DESEL clocks the controller is idle again
// (stop then start is a seek).
//
// mode  opcode (AW=24 / AW=32)  cmd-addr-data lanes  mode cycles  typical dummy
//  0    0x03 / 0x13             1-1-1                0            0    (<= ~50 MHz parts)
//  1    0x0B / 0x0C             1-1-1                0            8
//  2    0x6B / 0x6C             1-1-4                0            8    (needs QE = 1)
//  3    0xEB / 0xEC             1-4-4                2            4    (needs QE = 1)
//  4    0x8B / 0x7C             1-1-8                0            8    (octal parts, IOW = 8)
//  5    0xCB / 0xCC             1-8-8                0            16   (octal parts, IOW = 8)
// The number of dummy cycles is the `dummy` input (it depends on the part and the speed).
// The controller never writes the flash: QE / octal configuration is done when it is programmed.
//
// SCK = clk / (2 * HALF). Data is launched on the falling edge and the flash is sampled on
// the rising edge, plus rx_dly clk cycles to absorb the pad and board round trip.
// Pins: IO0 = MOSI, IO1 = MISO in single mode. IO2/IO3 (WP#/HOLD#) are driven high during
// the one-lane phases of a read.
module spi_flash_reader #(
    parameter IOW   = 4,      // 4 = quad, 8 = octal
    parameter AW    = 24,     // 24 or 32 address bits
    parameter HALF  = 1,      // clk cycles per SCK half period
    parameter DESEL = 4       // clk cycles CS# stays high after a stop
) (
    input  wire           clk,
    input  wire           rst_n,
    input  wire           start,
    input  wire           stop,
    input  wire [AW-1:0]  addr,
    input  wire [2:0]     mode,
    input  wire [4:0]     dummy,
    input  wire [1:0]     rx_dly,
    output wire           busy,
    output wire [7:0]     out_data,
    output wire           out_valid,
    input  wire           out_ready,
    output reg            sck_o,
    output reg            cs_n_o,
    output wire [IOW-1:0] io_o,
    output wire [IOW-1:0] io_oe,
    input  wire [IOW-1:0] io_i
);
    localparam [2:0] SEG_CMD = 3'd0, SEG_ADDR = 3'd1, SEG_MODE = 3'd2, SEG_DUMMY = 3'd3, SEG_DATA = 3'd4;
    localparam [1:0] ST_IDLE = 2'd0, ST_RUN = 2'd1, ST_STOP = 2'd2;
    localparam TW = AW + 8;
    localparam HW = (HALF > 1) ? $clog2(HALF) : 1;

    reg [1:0]    st;
    reg [2:0]    seg;
    reg [2:0]    mode_r;
    reg [4:0]    dummy_r;
    reg          ph;                 // 0 = SCK low half, 1 = high half
    reg [HW-1:0] hcnt;
    reg [5:0]    ncyc;               // cycles left in this segment, minus one
    reg [TW-1:0] tx;
    reg          committed;          // the byte being clocked has been counted in `pending`
    reg          stop_req;
    reg [7:0]    desel;

    // ---- mode decode
    function automatic [7:0] opcode(input [2:0] m);
        case (m)
            3'd0:    opcode = (AW == 32) ? 8'h13 : 8'h03;
            3'd1:    opcode = (AW == 32) ? 8'h0C : 8'h0B;
            3'd2:    opcode = (AW == 32) ? 8'h6C : 8'h6B;
            3'd3:    opcode = (AW == 32) ? 8'hEC : 8'hEB;
            3'd4:    opcode = (AW == 32) ? 8'h7C : 8'h8B;
            default: opcode = (AW == 32) ? 8'hCC : 8'hCB;
        endcase
    endfunction
    // lanes: 0 = one, 1 = four, 2 = eight
    wire [1:0] a_lanes = (mode_r == 3'd3) ? 2'd1 : (mode_r >= 3'd5) ? 2'd2 : 2'd0;
    wire [1:0] d_lanes = (mode_r <= 3'd1) ? 2'd0 : (mode_r <= 3'd3) ? 2'd1 : 2'd2;
    wire       has_mode = (mode_r == 3'd3);
    wire [5:0] addr_cyc = (a_lanes == 2'd0) ? AW - 1 : (a_lanes == 2'd1) ? AW / 4 - 1 : AW / 8 - 1;
    wire [5:0] data_cyc = (d_lanes == 2'd0) ? 6'd7 : (d_lanes == 2'd1) ? 6'd1 : 6'd0;

    // ---- pins (decoded from registers that only change on SCK falling edges)
    wire [7:0] top8 = tx[TW-1:TW-8];
    reg  [7:0] o8, oe8;
    always @* begin
        o8  = 8'hFF;
        oe8 = 8'h00;
        if (st == ST_RUN) begin
            case (seg)
                SEG_CMD: begin
                    o8  = {4'hF, 2'b11, 1'b1, top8[7]};
                    oe8 = 8'b0000_1101;
                end
                SEG_ADDR, SEG_MODE: begin
                    if (a_lanes == 2'd0) begin
                        o8  = {4'hF, 2'b11, 1'b1, top8[7]};
                        oe8 = 8'b0000_1101;
                    end else if (a_lanes == 2'd1) begin
                        o8  = {4'hF, top8[7:4]};
                        oe8 = 8'b0000_1111;
                    end else begin
                        o8  = top8;
                        oe8 = 8'hFF;
                    end
                end
                default: begin
                    o8  = 8'hFF;
                    oe8 = 8'h00;
                end
            endcase
        end
    end
    assign io_o  = o8[IOW-1:0];
    assign io_oe = oe8[IOW-1:0];

    // ---- receive side: strobes delayed by rx_dly, shift register, 4-byte FIFO
    localparam DEPTH = 4;
    reg  [3:0] stb_p, last_p;
    wire       rx_now, rx_last_now;
    wire       stb_s  = (rx_dly == 2'd0) ? rx_now      : stb_p[rx_dly - 2'd1];
    wire       last_s = (rx_dly == 2'd0) ? rx_last_now : last_p[rx_dly - 2'd1];
    reg  [7:0] sh;
    reg  [7:0] fifo [0:DEPTH-1];
    reg  [1:0] wp, rp;
    reg  [2:0] count;
    reg  [2:0] pending;
    wire       flush = (st == ST_STOP);
    wire [7:0] io8;
    generate
        if (IOW >= 8) begin : g_oct
            assign io8 = io_i[7:0];
        end else begin : g_quad
            assign io8 = {4'hF, io_i[3:0]};
        end
    endgenerate
    wire [7:0] sh_n  = (d_lanes == 2'd0) ? {sh[6:0], io_i[1]} :
                       (d_lanes == 2'd1) ? {sh[3:0], io_i[3:0]} : io8;
    wire       push  = stb_s & last_s & ~flush;
    wire       pop   = out_valid & out_ready;
    wire       space = ({1'b0, count} + {1'b0, pending}) < DEPTH;
    wire       commit;

    assign out_valid = (count != 3'd0) & ~flush;
    assign out_data  = fifo[rp];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stb_p   <= 4'd0;
            last_p  <= 4'd0;
            sh      <= 8'd0;
            wp      <= 2'd0;
            rp      <= 2'd0;
            count   <= 3'd0;
            pending <= 3'd0;
        end else if (flush) begin
            stb_p   <= 4'd0;
            last_p  <= 4'd0;
            wp      <= 2'd0;
            rp      <= 2'd0;
            count   <= 3'd0;
            pending <= 3'd0;
        end else begin
            stb_p  <= {stb_p[2:0], rx_now};
            last_p <= {last_p[2:0], rx_last_now};
            if (stb_s) sh <= sh_n;
            if (push) begin
                fifo[wp] <= sh_n;
                wp       <= wp + 2'd1;
            end
            if (pop) rp <= rp + 2'd1;
            count   <= count + {2'b00, push} - {2'b00, pop};
            pending <= pending + {2'b00, commit} - {2'b00, push};
        end
    end

    // ---- clock engine
    wire tick      = (hcnt == HALF - 1);
    wire in_data   = (st == ST_RUN) && (seg == SEG_DATA);
    wire new_byte  = in_data && !ph && (hcnt == {HW{1'b0}}) && !committed;
    wire stall     = new_byte && !space;
    assign commit      = new_byte && space && !(stop | stop_req);
    assign rx_now      = in_data && !ph && tick && !stall && (committed || commit);
    assign rx_last_now = rx_now && (ncyc == 6'd0);
    assign busy        = (st != ST_IDLE);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st        <= ST_IDLE;
            seg       <= SEG_CMD;
            mode_r    <= 3'd0;
            dummy_r   <= 5'd0;
            ph        <= 1'b0;
            hcnt      <= {HW{1'b0}};
            ncyc      <= 6'd0;
            tx        <= {TW{1'b0}};
            committed <= 1'b0;
            stop_req  <= 1'b0;
            desel     <= 8'd0;
            sck_o     <= 1'b0;
            cs_n_o    <= 1'b1;
        end else begin
            case (st)
                ST_IDLE: begin
                    sck_o    <= 1'b0;
                    cs_n_o   <= 1'b1;
                    stop_req <= 1'b0;
                    if (start) begin
                        st        <= ST_RUN;
                        seg       <= SEG_CMD;
                        mode_r    <= mode;
                        dummy_r   <= dummy;
                        tx        <= {opcode(mode), addr};
                        ncyc      <= 6'd7;
                        ph        <= 1'b0;
                        hcnt      <= {HW{1'b0}};
                        committed <= 1'b0;
                        cs_n_o    <= 1'b0;
                    end
                end
                ST_RUN: begin
                    if (stop) stop_req <= 1'b1;
                    if (!ph) begin
                        // low half
                        if ((stop | stop_req) && (stall || (new_byte && !commit))) begin
                            st    <= ST_STOP;
                            desel <= DESEL - 1;
                        end else if (!stall) begin
                            if (commit) committed <= 1'b1;
                            if (tick) begin
                                ph    <= 1'b1;
                                sck_o <= 1'b1;
                                hcnt  <= {HW{1'b0}};
                            end else begin
                                hcnt <= hcnt + 1'b1;
                            end
                        end
                    end else begin
                        // high half
                        if (tick) begin
                            ph    <= 1'b0;
                            sck_o <= 1'b0;
                            hcnt  <= {HW{1'b0}};
                            if (seg == SEG_DATA) begin
                                if (ncyc == 6'd0) begin
                                    ncyc      <= data_cyc;
                                    committed <= 1'b0;
                                end else begin
                                    ncyc <= ncyc - 6'd1;
                                end
                            end else if (ncyc != 6'd0) begin
                                ncyc <= ncyc - 6'd1;
                                if (seg == SEG_CMD || a_lanes == 2'd0) tx <= {tx[TW-2:0], 1'b1};
                                else if (a_lanes == 2'd1)              tx <= {tx[TW-5:0], 4'hF};
                                else                                   tx <= {tx[TW-9:0], 8'hFF};
                            end else begin
                                // end of a segment: shift once more, then pick the next non-empty one
                                if (seg == SEG_CMD || a_lanes == 2'd0) tx <= {tx[TW-2:0], 1'b1};
                                else if (a_lanes == 2'd1)              tx <= {tx[TW-5:0], 4'hF};
                                else                                   tx <= {tx[TW-9:0], 8'hFF};
                                if (seg == SEG_CMD) begin
                                    seg  <= SEG_ADDR;
                                    ncyc <= addr_cyc;
                                end else if (seg == SEG_ADDR && has_mode) begin
                                    seg  <= SEG_MODE;
                                    ncyc <= 6'd1;
                                end else if (seg != SEG_DUMMY && dummy_r != 5'd0) begin
                                    seg  <= SEG_DUMMY;
                                    ncyc <= {1'b0, dummy_r} - 6'd1;
                                end else begin
                                    seg  <= SEG_DATA;
                                    ncyc <= data_cyc;
                                end
                            end
                        end else begin
                            hcnt <= hcnt + 1'b1;
                        end
                    end
                end
                default: begin  // ST_STOP
                    sck_o    <= 1'b0;
                    cs_n_o   <= 1'b1;
                    stop_req <= 1'b0;
                    if (desel == 8'd0) st <= ST_IDLE;
                    else               desel <= desel - 8'd1;
                end
            endcase
        end
    end
endmodule
