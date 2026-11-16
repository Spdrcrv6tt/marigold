// Event frames (docs/config-blob.md "Events and host message", docs/architecture.md sections 8.4 and 8.7), island
// clock domain. go queues one frame: 0xA5, type, action (COMMAND only, else 0), n_values, then for each slot k of
// slot_mask (COMMAND only) k and its value as int32 little endian, confidence (COMMAND and NOMATCH, else 0), and
// CRC-8 (polynomial 0x07, initial 0, SMBus PEC) over every byte before it. Frames wait in a 2-frame FIFO (a frame
// that finds it full is dropped, `dropped` set) and leave by I2C reads of MSG_DATA (pop) or, with host = 2, on the
// UART. irq_o (when irq_en) is high while a frame is pending or the UART is still sending.
module mg_msg #(
    parameter UDIV = 26                  // UART bit time = UDIV + 1 clk_fe clocks (113.8 kbaud at 3.072 MHz)
) (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         go,
    input  wire [1:0]   ev,
    input  wire [7:0]   action,
    input  wire [7:0]   slot_mask,
    input  wire [255:0] vals,
    input  wire [7:0]   conf,
    output wire         busy,
    input  wire [7:0]   host,
    input  wire         irq_en,
    output wire [5:0]   head_len,         // bytes of the oldest frame (0 = none)
    output wire [7:0]   head_byte,        // its next byte
    input  wire         pop,              // I2C read of that byte
    input  wire         drop_all,
    output reg          dropped,
    input  wire         clr_dropped,
    output wire         tx_o,
    output wire         irq_o
);
    reg  [7:0]  fb [0:127];               // frame s at bytes 64 s .. 64 s + 45
    reg  [5:0]  flen [0:1];
    reg         hd, tl;                   // head and tail frame
    reg  [1:0]  cnt;
    reg  [5:0]  rp;                       // read position in the head frame

    // ---- builder
    localparam B_IDLE = 2'd0, B_HDR = 2'd1, B_VAL = 2'd2, B_END = 2'd3;
    reg  [1:0]  bs;
    reg  [5:0]  wp;
    reg  [2:0]  bk;                       // slot
    reg  [2:0]  bj;                       // byte within the slot record (0 = k, 1..4 = value)
    reg  [7:0]  crc;
    reg  [1:0]  e_ev;
    reg  [7:0]  e_act, e_mask, e_conf;
    reg  [255:0] e_vals;
    assign busy = (bs != B_IDLE);

    function [7:0] crc8(input [7:0] c, input [7:0] b);
        integer k;
        reg [7:0] x;
        begin
            x = c ^ b;
            for (k = 0; k < 8; k = k + 1) x = x[7] ? ((x << 1) ^ 8'h07) : (x << 1);
            crc8 = x;
        end
    endfunction

    wire [3:0]  nval = e_mask[0] + e_mask[1] + e_mask[2] + e_mask[3] + e_mask[4] + e_mask[5] + e_mask[6] + e_mask[7];
    wire [31:0] v32  = e_vals[32*bk +: 32];
    wire [7:0]  vbyte = (bj == 3'd0) ? {5'd0, bk} : v32[8*(bj-3'd1) +: 8];
    wire [7:0]  hbyte = (wp == 6'd0) ? 8'hA5 : (wp == 6'd1) ? {6'd0, e_ev} : (wp == 6'd2) ? e_act : {4'd0, nval};
    reg  [7:0]  wb;
    reg         wr;
    always @(*) begin
        wr = 1'b0; wb = 8'd0;
        case (bs)
            B_HDR: begin wr = 1'b1; wb = hbyte; end
            B_VAL: if (e_mask[bk]) begin wr = 1'b1; wb = vbyte; end
            B_END: begin wr = 1'b1; wb = (bj == 3'd0) ? e_conf : crc; end
            default: ;
        endcase
    end

    // ---- UART
    wire        gclk;
    wire        u_ready;
    wire        u_send = (host == 8'd2) && (cnt != 2'd0) && u_ready && !busy;
    uart_tx #(.DW(8)) u_tx (.clk(gclk), .rst_n(rst_n), .div(UDIV[7:0]), .in_data(head_byte), .in_valid(u_send),
                            .in_ready(u_ready), .tx_o(tx_o));
    wire        rd = (pop && host != 8'd2) || u_send;

    assign head_len  = (cnt != 2'd0) ? flen[hd] : 6'd0;
    assign head_byte = (cnt != 2'd0) ? fb[{hd, rp}] : 8'd0;
    assign irq_o = irq_en && ((cnt != 2'd0) || !u_ready);

    wire gen = go || busy || (cnt != 2'd0) || !u_ready || pop || drop_all || clr_dropped;
    clk_gate u_cg (.clk(clk), .en(gen), .gclk(gclk));

    always @(posedge gclk or negedge rst_n) begin
        if (!rst_n) begin
            hd <= 1'b0; tl <= 1'b0; cnt <= 2'd0; rp <= 6'd0; bs <= B_IDLE; wp <= 6'd0; bk <= 3'd0; bj <= 3'd0;
            crc <= 8'd0; e_ev <= 2'd0; e_act <= 8'd0; e_mask <= 8'd0; e_conf <= 8'd0; e_vals <= 256'd0; dropped <= 1'b0;
            flen[0] <= 6'd0; flen[1] <= 6'd0;
        end else begin
            if (clr_dropped) dropped <= 1'b0;
            // read side
            if (drop_all) begin
                cnt <= 2'd0; rp <= 6'd0; hd <= tl;
            end else if (rd && cnt != 2'd0) begin
                if (rp == flen[hd] - 6'd1) begin rp <= 6'd0; hd <= ~hd; cnt <= cnt - 2'd1; end
                else rp <= rp + 6'd1;
            end
            // builder
            if (wr) begin fb[{tl, wp}] <= wb; crc <= crc8(crc, wb); wp <= wp + 6'd1; end
            case (bs)
                B_IDLE: if (go) begin
                    if ((cnt == 2'd2) && !(rd && rp == flen[hd] - 6'd1)) dropped <= 1'b1;
                    else begin
                        bs <= B_HDR; wp <= 6'd0; crc <= 8'd0; e_ev <= ev;
                        e_act  <= (ev == 2'd0) ? action : 8'd0;
                        e_mask <= (ev == 2'd0) ? slot_mask : 8'd0;
                        e_conf <= (ev == 2'd0 || ev == 2'd2) ? conf : 8'd0;
                        e_vals <= vals;
                    end
                end
                B_HDR: if (wp == 6'd3) begin bs <= B_VAL; bk <= 3'd0; bj <= 3'd0; end
                B_VAL: begin
                    if (!e_mask[bk] || bj == 3'd4) begin
                        bj <= 3'd0;
                        if (bk == 3'd7) bs <= B_END;
                        bk <= bk + 3'd1;
                    end else bj <= bj + 3'd1;
                end
                B_END: begin
                    if (bj == 3'd0) bj <= 3'd1;
                    else begin
                        bs <= B_IDLE; flen[tl] <= wp + 6'd1; tl <= ~tl;
                        cnt <= (rd && cnt != 2'd0 && rp == flen[hd] - 6'd1 && !drop_all) ? cnt : (drop_all ? 2'd1 : cnt + 2'd1);
                    end
                end
            endcase
        end
    end
endmodule
