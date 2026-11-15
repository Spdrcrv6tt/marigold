// Behavioural SPI NOR flash for the read controller tests (not synthesised, written in-house).
// Understands the read opcodes in spi_flash_reader.sv with a fixed dummy count per opcode,
// samples on SCK rising, drives data t_clqv after SCK falling. Contents are a hash of the
// address, pat() below; rtl/tb/test_flash.py computes the same.
`timescale 1ps/1ps
module flash_model #(
    parameter IOW = 4,
    parameter AW  = 24
) (
    input  wire           sck,
    input  wire           cs_n,
    input  wire [IOW-1:0] io_in,
    output reg  [IOW-1:0] io_out,
    output reg  [IOW-1:0] io_oe,
    input  wire [31:0]    t_clqv,       // ps, SCK low to output valid
    output reg            bad_cmd,
    output reg  [7:0]     last_cmd,
    output reg  [31:0]    n_selects
);
    localparam P_CMD = 0, P_ADDR = 1, P_MODE = 2, P_DUMMY = 3, P_DATA = 4, P_IGNORE = 5;

    function automatic [7:0] pat(input [31:0] a);
        reg [31:0] x;
        begin
            x   = a * 32'h9E3779B1;
            pat = x[31:24] ^ a[7:0];
        end
    endfunction

    wire [7:0] in8 = io_in;
    integer    phase, cnt, a_lanes, d_lanes, mode_cyc, dummy_cyc, chunk;
    reg [7:0]  cmd;
    reg [31:0] addr;
    reg [7:0]  cur;
    reg [7:0]  v;

    initial begin
        io_out = {IOW{1'b1}}; io_oe = {IOW{1'b0}}; bad_cmd = 0; last_cmd = 0; n_selects = 0;
        phase = P_CMD; cnt = 0; a_lanes = 1; d_lanes = 1; mode_cyc = 0; dummy_cyc = 0; chunk = 0;
        cmd = 0; addr = 0; cur = 0; v = 0;
    end

    always @(negedge cs_n) begin
        phase = P_CMD; cnt = 0; cmd = 0; addr = 0; chunk = 0;
        n_selects = n_selects + 1;
    end

    always @(posedge cs_n) begin
        phase = P_IGNORE;
        io_oe <= {IOW{1'b0}};
    end

    always @(posedge sck) if (!cs_n) begin
        case (phase)
            P_CMD: begin
                cmd = {cmd[6:0], io_in[0]};
                cnt = cnt + 1;
                if (cnt == 8) begin
                    last_cmd = cmd;
                    cnt = 0; mode_cyc = 0; phase = P_ADDR;
                    case (cmd)
                        8'h03, 8'h13: begin a_lanes = 1; d_lanes = 1; dummy_cyc = 0;  end
                        8'h0B, 8'h0C: begin a_lanes = 1; d_lanes = 1; dummy_cyc = 8;  end
                        8'h6B, 8'h6C: begin a_lanes = 1; d_lanes = 4; dummy_cyc = 8;  end
                        8'hEB, 8'hEC: begin a_lanes = 4; d_lanes = 4; dummy_cyc = 4; mode_cyc = 2; end
                        8'h8B, 8'h7C: begin a_lanes = 1; d_lanes = 8; dummy_cyc = 8;  end
                        8'hCB, 8'hCC: begin a_lanes = 8; d_lanes = 8; dummy_cyc = 16; end
                        default:      begin bad_cmd = 1; phase = P_IGNORE; end
                    endcase
                    if (d_lanes > IOW || a_lanes > IOW) begin bad_cmd = 1; phase = P_IGNORE; end
                    // 3-byte opcodes on a 32-bit build (and the reverse) are a test bench error
                    if ((AW == 32) != (cmd == 8'h13 || cmd == 8'h0C || cmd == 8'h6C || cmd == 8'hEC || cmd == 8'h7C || cmd == 8'hCC)) begin
                        bad_cmd = 1; phase = P_IGNORE;
                    end
                end
            end
            P_ADDR: begin
                if (a_lanes == 1)      addr = {addr[30:0], io_in[0]};
                else if (a_lanes == 4) addr = {addr[27:0], io_in[3:0]};
                else                   addr = {addr[23:0], in8};
                cnt = cnt + a_lanes;
                if (cnt == AW) begin
                    cnt = 0;
                    phase = (mode_cyc != 0) ? P_MODE : (dummy_cyc != 0) ? P_DUMMY : P_DATA;
                end
            end
            P_MODE: begin
                cnt = cnt + 1;
                if (cnt == mode_cyc) begin cnt = 0; phase = (dummy_cyc != 0) ? P_DUMMY : P_DATA; end
            end
            P_DUMMY: begin
                cnt = cnt + 1;
                if (cnt == dummy_cyc) begin cnt = 0; phase = P_DATA; end
            end
            default: ;
        endcase
    end

    always @(negedge sck) if (!cs_n && phase == P_DATA) begin
        if (chunk == 0) cur = pat(addr);
        if (d_lanes == 1) begin
            v = {IOW{1'b1}};
            v[1] = cur[7 - chunk];
            io_out <= #(t_clqv) v[IOW-1:0];
            io_oe  <= #(t_clqv) {{(IOW-2){1'b0}}, 2'b10};
            chunk = chunk + 1;
            if (chunk == 8) begin chunk = 0; addr = addr + 1; end
        end else if (d_lanes == 4) begin
            v = 8'hFF;
            v[3:0] = (chunk == 0) ? cur[7:4] : cur[3:0];
            io_out <= #(t_clqv) v[IOW-1:0];
            io_oe  <= #(t_clqv) {{(IOW-4){1'b0}}, 4'hF};
            chunk = chunk + 1;
            if (chunk == 2) begin chunk = 0; addr = addr + 1; end
        end else begin
            v = cur;
            io_out <= #(t_clqv) v[IOW-1:0];
            io_oe  <= #(t_clqv) {IOW{1'b1}};
            addr = addr + 1;
        end
    end
endmodule
