// I2C target (docs/architecture.md section 6), 7-bit address, standard mode (100 kHz), oversampled on clk_fe
// (3.072 MHz) through two-flop synchronisers; no clock stretching. SDA changes 2 to 3 clk_fe periods (651 to 977
// ns) after SCL falls: inside standard mode's tVD;DAT (3.45 us) with a 1 us rise, but not fast mode's 0.9 us once
// the bus rise time is counted (review R7; test_ctl.py i2c_timing measures both).
// Register protocol: the first byte written after the address sets the pointer; further written bytes go to the
// pointer (wr_stb); reads return the pointer's register (rd_data, combinational from ptr; rd_stb when a byte is
// taken). The pointer increments after each byte except on the stream registers S0 and S1.
module mg_i2c #(
    parameter [7:0] S0 = 8'h03,          // CFG_DATA
    parameter [7:0] S1 = 8'h05           // MSG_DATA
) (
    input  wire       clk,
    input  wire       rst_n,
    input  wire       scl_i,
    input  wire       sda_i,
    output reg        sda_oe,
    input  wire [6:0] addr,
    output reg  [7:0] ptr,
    output reg        wr_stb,
    output reg  [7:0] wr_reg,            // with wr_stb: the register written
    output reg  [7:0] wr_data,
    output reg        rd_stb,
    output reg  [7:0] rd_reg,            // with rd_stb: the register read
    input  wire [7:0] rd_data
);
    reg  [1:0] scl_s, sda_s;
    reg        scl_p, sda_p;
    wire       scl = scl_s[1], sda = sda_s[1];
    wire       start = scl && scl_p && sda_p && !sda;
    wire       stop  = scl && scl_p && !sda_p && sda;
    wire       rise  = scl && !scl_p;
    wire       fall  = !scl && scl_p;

    localparam T_IDLE = 3'd0, T_ADDR = 3'd1, T_AACK = 3'd2, T_WR = 3'd3, T_WACK = 3'd4, T_RD = 3'd5, T_RACK = 3'd6;
    reg  [2:0] st;
    reg  [3:0] nb;
    reg  [7:0] sh, tx;
    reg        rw, first, nack;
    wire       stream = (ptr == S0) || (ptr == S1);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            scl_s <= 2'b11; sda_s <= 2'b11; scl_p <= 1'b1; sda_p <= 1'b1; st <= T_IDLE; nb <= 4'd0; sh <= 8'd0;
            tx <= 8'd0; rw <= 1'b0; first <= 1'b0; nack <= 1'b0; sda_oe <= 1'b0; ptr <= 8'd0; wr_stb <= 1'b0;
            wr_reg <= 8'd0; wr_data <= 8'd0; rd_stb <= 1'b0; rd_reg <= 8'd0;
        end else begin
            scl_s <= {scl_s[0], scl_i}; sda_s <= {sda_s[0], sda_i}; scl_p <= scl; sda_p <= sda;
            wr_stb <= 1'b0; rd_stb <= 1'b0;
            if (start) begin
                st <= T_ADDR; nb <= 4'd0; sda_oe <= 1'b0;
            end else if (stop) begin
                st <= T_IDLE; sda_oe <= 1'b0;
            end else case (st)
                T_ADDR: begin
                    if (rise) begin sh <= {sh[6:0], sda}; nb <= nb + 4'd1; end
                    if (fall && nb == 4'd8) begin
                        if (sh[7:1] == addr) begin sda_oe <= 1'b1; rw <= sh[0]; st <= T_AACK; end
                        else st <= T_IDLE;
                    end
                end
                T_AACK: if (fall) begin
                    nb <= 4'd0;
                    if (!rw) begin sda_oe <= 1'b0; first <= 1'b1; st <= T_WR; end
                    else begin
                        tx <= rd_data; rd_stb <= 1'b1; rd_reg <= ptr; sda_oe <= ~rd_data[7]; st <= T_RD;
                        if (!stream) ptr <= ptr + 8'd1;
                    end
                end
                T_WR: begin
                    if (rise) begin sh <= {sh[6:0], sda}; nb <= nb + 4'd1; end
                    if (fall && nb == 4'd8) begin
                        sda_oe <= 1'b1; st <= T_WACK;
                        if (first) begin ptr <= sh; first <= 1'b0; end
                        else begin
                            wr_stb <= 1'b1; wr_reg <= ptr; wr_data <= sh;
                            if (!stream) ptr <= ptr + 8'd1;
                        end
                    end
                end
                T_WACK: if (fall) begin
                    sda_oe <= 1'b0; nb <= 4'd0; st <= T_WR;
                end
                T_RD: begin
                    if (rise) nb <= nb + 4'd1;
                    if (fall) begin
                        if (nb == 4'd8) begin sda_oe <= 1'b0; st <= T_RACK; end
                        else sda_oe <= ~tx[3'd7 - nb[2:0]];
                    end
                end
                T_RACK: begin
                    if (rise) nack <= sda;
                    if (fall) begin
                        nb <= 4'd0;
                        if (nack) st <= T_IDLE;
                        else begin
                            tx <= rd_data; rd_stb <= 1'b1; rd_reg <= ptr; sda_oe <= ~rd_data[7]; st <= T_RD;
                            if (!stream) ptr <= ptr + 8'd1;
                        end
                    end
                end
                default: ;
            endcase
        end
    end
endmodule
