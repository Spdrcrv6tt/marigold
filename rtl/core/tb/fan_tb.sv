// Unit test wrapper: imh_fan (fan-out array) + activation RAM. The harness (test_fan.cpp) owns the RAM port while
// the array is idle and plays the weight source itself.
module fan_tb #(
    parameter LANES = 16,            // RAM word elements (the core's LANES)
    parameter RAM_BYTES = 32768
) (
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 reg_we,
    input  wire [4:0]           reg_idx,
    input  wire [31:0]          reg_wdata,
    input  wire                 go,
    output wire                 done,
    output wire [31:0]          argidx,
    output wire [31:0]          argval,
    output wire [255:0]         argidx_all,
    input  wire                 h_en,
    input  wire                 h_we,
    input  wire [31:0]          h_addr,
    input  wire [LANES*16-1:0]  h_wdata,
    output wire [LANES*16-1:0]  h_rdata,
    output wire                 seek_valid,
    output wire [31:0]          seek_addr,
    input  wire [8*LANES-1:0]   ws_data,
    input  wire [5:0]           ws_avail,
    output wire [5:0]           ws_take
);
    localparam NB = LANES * 2;
    wire               busy, f_en;
    wire [NB-1:0]      f_we;
    wire [31:0]        f_addr;
    wire [LANES*16-1:0] f_wdata;
    imh_fan #(.L(32), .CMAX(8), .RWL(LANES), .WB(LANES)) u_fan (
        .clk(clk), .rst_n(rst_n), .reg_we(reg_we), .reg_idx(reg_idx), .reg_wdata(reg_wdata), .go(go),
        .busy(busy), .done(done), .argidx(argidx), .argval(argval), .argidx_all(argidx_all),
        .m_en(f_en), .m_we(f_we), .m_addr(f_addr), .m_wdata(f_wdata), .m_rdata(h_rdata),
        .seek_valid(seek_valid), .seek_addr(seek_addr), .ws_data(ws_data), .ws_avail(busy ? ws_avail : 6'd0), .ws_take(ws_take));
    imh_ram #(.LANES(LANES), .BYTES(RAM_BYTES)) ram (
        .clk(clk), .en(busy ? f_en : h_en), .we(busy ? f_we : {NB{h_we}}),
        .addr(busy ? f_addr : h_addr), .wdata(busy ? f_wdata : h_wdata), .rdata(h_rdata));
endmodule
