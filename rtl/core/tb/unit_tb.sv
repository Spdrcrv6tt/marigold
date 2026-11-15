// Unit test wrapper: imh_mac + imh_vec + activation RAM. The harness (test_mac.cpp) owns the RAM port while
// both units are idle and plays the weight source itself.
module unit_tb #(
    parameter LANES = 16,
    parameter RAM_BYTES = 16384
) (
    clk, rst_n, reg_we_m, reg_we_v, reg_idx, reg_wdata, go_m, go_v, done_m, done_v, argidx, argval, satflag, s_acc, q_acc, o_mean, o_rec, o_sh1,
    h_en, h_we, h_addr, h_wdata, h_rdata,
    seek_valid, seek_addr, ws_data, ws_avail, ws_take
);
    localparam W = LANES * 16;
    localparam NB = LANES * 2;
    input  wire          clk;
    input  wire          rst_n;
    input  wire          reg_we_m;
    input  wire          reg_we_v;
    input  wire [4:0]    reg_idx;
    input  wire [31:0]   reg_wdata;
    input  wire          go_m;
    input  wire          go_v;
    output wire          done_m;
    output wire          done_v;
    output wire [31:0]   argidx;
    output wire [31:0]   argval;
    output wire          satflag;
    output wire [63:0]   s_acc;
    output wire [63:0]   q_acc;
    output wire [31:0]   o_mean;
    output wire [31:0]   o_rec;
    output wire [31:0]   o_sh1;
    input  wire          h_en;
    input  wire          h_we;
    input  wire [31:0]   h_addr;
    input  wire [W-1:0]  h_wdata;
    output wire [W-1:0]  h_rdata;
    output wire          seek_valid;
    output wire [31:0]   seek_addr;
    input  wire [8*LANES-1:0] ws_data;
    input  wire [5:0]    ws_avail;
    output wire [5:0]    ws_take;

    wire          busy_m, busy_v, m_en, v_en, sk_m, sk_v;
    wire [NB-1:0] m_we, v_we;
    wire [31:0]   m_addr, v_addr, sa_m, sa_v;
    wire [W-1:0]  m_wdata, v_wdata;
    wire [5:0]    tk_m, tk_v;
    imh_mac #(.LANES(LANES)) u_mac (
        .clk(clk), .rst_n(rst_n), .reg_we(reg_we_m), .reg_idx(reg_idx), .reg_wdata(reg_wdata), .go(go_m),
        .busy(busy_m), .done(done_m), .argidx(argidx), .argval(argval), .satflag(satflag),
        .m_en(m_en), .m_we(m_we), .m_addr(m_addr), .m_wdata(m_wdata), .m_rdata(h_rdata),
        .seek_valid(sk_m), .seek_addr(sa_m), .ws_data(ws_data), .ws_avail(busy_m ? ws_avail : 6'd0), .ws_take(tk_m));
    imh_vec #(.LANES(LANES)) u_vec (
        .clk(clk), .rst_n(rst_n), .reg_we(reg_we_v), .reg_idx(reg_idx), .reg_wdata(reg_wdata), .go(go_v),
        .busy(busy_v), .done(done_v), .s_acc(s_acc), .q_acc(q_acc), .o_mean(o_mean), .o_rec(o_rec), .o_sh1(o_sh1),
        .m_en(v_en), .m_we(v_we), .m_addr(v_addr), .m_wdata(v_wdata), .m_rdata(h_rdata),
        .seek_valid(sk_v), .seek_addr(sa_v), .ws_data(ws_data), .ws_avail(busy_v ? ws_avail : 6'd0), .ws_take(tk_v));
    assign seek_valid = sk_m | sk_v;
    assign seek_addr  = sk_m ? sa_m : sa_v;
    assign ws_take    = busy_m ? tk_m : busy_v ? tk_v : 6'd0;
    imh_ram #(.LANES(LANES), .BYTES(RAM_BYTES)) ram (
        .clk(clk), .en(busy_m ? m_en : busy_v ? v_en : h_en), .we(busy_m ? m_we : busy_v ? v_we : {NB{h_we}}),
        .addr(busy_m ? m_addr : busy_v ? v_addr : h_addr), .wdata(busy_m ? m_wdata : busy_v ? v_wdata : h_wdata), .rdata(h_rdata));
endmodule
