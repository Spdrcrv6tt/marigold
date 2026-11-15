// Config loader (docs/architecture.md sections 3 and 8.6): takes the blob as a byte stream, writes it to config
// SRAM, keeps the 32-byte header, and on commit checks magic, version, field ranges, length and CRC-32 (IEEE,
// zlib.crc32) in that order. The CRC runs on the fly four bytes behind the stream, so at commit it covers every
// byte but the last four, which are still in the delay line and are the stored CRC. cfg_valid only after a pass.
module mg_cfg_load (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        begin_i,           // one clock: forget the config, expect a new blob from byte 0
    input  wire        commit_i,          // one clock: check what was received
    input  wire        in_valid,
    input  wire [7:0]  in_data,
    output wire        we,
    output wire [14:0] waddr,
    output wire [7:0]  wdata,
    output reg         valid,
    output reg  [3:0]  err,               // 0 none, 1 magic, 2 version, 3 field range, 4 length, 5 CRC
    output reg  [15:0] cnt,               // bytes received (saturates at 0xFFFF)
    // header fields and table bases, stable while valid
    output wire [15:0] D,
    output wire [15:0] N,
    output wire [7:0]  n_actions,
    output wire [7:0]  n_slots,
    output wire [7:0]  n_gpio,
    output wire [7:0]  wake_len,
    output wire [7:0]  host,
    output wire [7:0]  i2c_addr,
    output wire [31:0] reject_thr,
    output wire [31:0] margin,
    output wire [15:0] wake_sens,
    output wire [15:0] flags,
    output wire [7:0]  conf_sh,
    output reg  [14:0] vec_base,
    output reg  [14:0] act_base,
    output reg  [14:0] slot_base,
    output reg  [14:0] gpio_base,
    output reg  [14:0] wake_base
);
    reg  [7:0]  hdr [0:31];
    reg  [31:0] crc, dl;

    function [31:0] crc_byte(input [31:0] c, input [7:0] b);
        integer k;
        reg [31:0] x;
        begin
            x = c ^ {24'd0, b};
            for (k = 0; k < 8; k = k + 1) x = x[0] ? ((x >> 1) ^ 32'hEDB88320) : (x >> 1);
            crc_byte = x;
        end
    endfunction

    assign D          = {hdr[7], hdr[6]};
    assign N          = {hdr[9], hdr[8]};
    assign n_actions  = hdr[10];
    assign n_slots    = hdr[11];
    assign n_gpio     = hdr[12];
    assign wake_len   = hdr[13];
    assign host       = hdr[14];
    assign i2c_addr   = hdr[15];
    assign reject_thr = {hdr[19], hdr[18], hdr[17], hdr[16]};
    assign margin     = {hdr[23], hdr[22], hdr[21], hdr[20]};
    assign wake_sens  = {hdr[25], hdr[24]};
    assign flags      = {hdr[27], hdr[26]};
    assign conf_sh    = hdr[28];
    wire [15:0] ver   = {hdr[5], hdr[4]};

    assign we    = in_valid && (cnt < 16'd32768) && !begin_i;
    assign waddr = cnt[14:0];
    assign wdata = in_data;

    // expected length and table bases (used at commit)
    wire [31:0] nd    = N * D;
    wire [31:0] vb    = 32'h20 + {N, 2'b00};
    wire [31:0] ab    = vb + nd;
    wire [31:0] sb    = ab + {n_actions, 2'b00};
    wire [31:0] gb    = sb + n_slots * 12;
    wire [31:0] wb    = gb + {n_gpio, 3'b000};
    wire [31:0] len   = wb + wake_len + 32'd4;
    wire        magic = (cnt >= 16'd32) && hdr[0] == "M" && hdr[1] == "G" && hdr[2] == "C" && hdr[3] == "0";
    wire        range = (n_actions >= 8'd1) && (n_actions <= 8'd64) && (n_slots <= 8'd8) && (n_gpio <= 8'd32) &&
                        (host <= 8'd2) && (D >= 16'd1) && (D <= 16'd256) && (N >= 16'd1);
    wire [3:0]  chk   = !magic ? 4'd1 : (ver != 16'd0) ? 4'd2 : !range ? 4'd3 :
                        ((len > 32'd32768) || (len != {16'd0, cnt})) ? 4'd4 : ((~crc) != dl) ? 4'd5 : 4'd0;

    wire gen = begin_i | commit_i | in_valid;
    wire gclk;
    clk_gate u_cg (.clk(clk), .en(gen), .gclk(gclk));

    integer i;
    always @(posedge gclk or negedge rst_n) begin
        if (!rst_n) begin
            valid <= 1'b0; err <= 4'd0; cnt <= 16'd0; crc <= 32'hFFFFFFFF; dl <= 32'd0;
            for (i = 0; i < 32; i = i + 1) hdr[i] <= 8'd0;
            vec_base <= 15'd0; act_base <= 15'd0; slot_base <= 15'd0; gpio_base <= 15'd0; wake_base <= 15'd0;
        end else if (begin_i) begin
            valid <= 1'b0; err <= 4'd0; cnt <= 16'd0; crc <= 32'hFFFFFFFF; dl <= 32'd0;
            for (i = 0; i < 32; i = i + 1) hdr[i] <= 8'd0;
        end else if (commit_i) begin
            valid <= (chk == 4'd0);
            err <= chk;
            vec_base <= vb[14:0]; act_base <= ab[14:0]; slot_base <= sb[14:0]; gpio_base <= gb[14:0];
            wake_base <= wb[14:0];
        end else if (in_valid) begin
            if (cnt < 16'd32) hdr[cnt[4:0]] <= in_data;
            if (cnt >= 16'd4) crc <= crc_byte(crc, dl[7:0]);
            dl <= {in_data, dl[31:8]};
            if (cnt != 16'hFFFF) cnt <= cnt + 16'd1;
            valid <= 1'b0;
        end
    end
endmodule
