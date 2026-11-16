// Marigold wake spotter (docs/golden-wake.md; interface docs/architecture.md section 3.1, rules agreed Nov 15).
// clk_fe domain (3.072 MHz). Sample n = 0 is the first in_valid with en = 1; while en = 0 all state clears.
// Network frame t covers samples [320t, 320t+560): it starts after sample 320t+559 lands in the 1024-sample ring,
// runs front end -> phone network -> keyword DP (about 45k of the 61,440 clocks per frame), and a fire for frame t
// raises `wake` together with the in_valid of sample k = 320t + 879 (the next frame's last sample).
// in_vad / in_raw are not used in v1 (no VAD power gating, no VAD qualifier). tpl_len is not needed: the template
// is self-describing (docs/golden-wake.md, section Template).
module mg_wake #(
    parameter MEMDIR    = "",
    parameter ROM_STALL = 0
) (
    input  wire               clk,
    input  wire               rst_n,
    input  wire               en,
    input  wire               in_valid,
    input  wire signed [15:0] in_data,
    input  wire               in_vad,
    input  wire               in_raw,
    input  wire [14:0]        tpl_addr,
    input  wire [7:0]         tpl_len,
    input  wire [15:0]        sens,
    output wire               cfg_req,
    output wire [14:0]        cfg_addr,
    input  wire               cfg_gnt,
    input  wire [31:0]        cfg_rdata,
    output wire               wake,
    output wire               busy,
    output wire               dbg_v,         // trace for the testbench (leave open in the chip)
    output wire [1:0]         dbg_k,         // 0 feature {frame b, band, v}, 1 logit {idx, z}, 2 best score, 3 overrun
    output wire [31:0]        dbg_d
);
    wire clr = !en;

    // ---- sample ring and frame schedule
    reg  [9:0] wptr, left, fbase;
    reg        fire_q, ovr;
    wire       acc  = en & in_valid;
    wire       trig = acc & (left == 10'd1);
    assign wake = trig & fire_q;
    wire        sr_re;
    wire [9:0]  sr_addr;
    wire [15:0] sr_rdata;
    wk_ram #(.WORDS(1024), .WIDTH(16)) u_ring (.clk(clk), .we(acc), .waddr(wptr), .wdata(in_data), .re(sr_re), .raddr(sr_addr), .rdata(sr_rdata));

    wire fe_busy, fe_done, nn_busy, nn_done, dp_busy, dp_done, dp_fire;
    reg  fe_start;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wptr <= 0; left <= 10'd560; fire_q <= 0; fe_start <= 0; fbase <= 0; ovr <= 0;
        end else if (clr) begin
            wptr <= 0; left <= 10'd560; fire_q <= 0; fe_start <= 0;
        end else begin
            fe_start <= 0;
            if (acc) begin
                wptr <= wptr + 1;
                left <= trig ? 10'd320 : left - 1;
                if (trig) begin
                    fe_start <= 1;
                    fbase <= wptr - 10'd559;
                    fire_q <= 0;
                    if (busy) ovr <= 1;
                end
            end
            if (dp_fire) fire_q <= 1;
        end
    end

    // ---- engines
    wire       x_we;  wire [5:0] x_addr;  wire [7:0] x_data;
    wire       fe_dv; wire [15:0] fe_dd;
    wk_fe #(.MEMDIR(MEMDIR)) u_fe (.clk(clk), .rst_n(rst_n), .clr(clr), .start(fe_start), .base(fbase), .busy(fe_busy), .done(fe_done),
        .sr_re(sr_re), .sr_addr(sr_addr), .sr_rdata(sr_rdata), .x_we(x_we), .x_addr(x_addr), .x_data(x_data), .dbg_v(fe_dv), .dbg_d(fe_dd));

    wire        rom_req, rom_gnt, rom_rvalid;
    wire [14:0] rom_addr;
    wire [15:0] rom_rdata;
    wk_rom #(.STALL(ROM_STALL), .MEMFILE({MEMDIR, "wake_rom.mem"})) u_rom (.clk(clk), .rst_n(rst_n), .rom_req(rom_req), .rom_addr(rom_addr),
        .rom_gnt(rom_gnt), .rom_rvalid(rom_rvalid), .rom_rdata(rom_rdata));

    wire [5:0]  z_idx;  wire signed [15:0] z_val, zmax;
    wire        nn_dv;  wire [21:0] nn_dd;
    wk_nn u_nn (.clk(clk), .rst_n(rst_n), .clr(clr), .start(fe_done), .busy(nn_busy), .done(nn_done),
        .x_we(x_we), .x_addr(x_addr), .x_data(x_data), .rom_req(rom_req), .rom_addr(rom_addr), .rom_gnt(rom_gnt),
        .rom_rvalid(rom_rvalid), .rom_rdata(rom_rdata), .z_idx(z_idx), .z_val(z_val), .zmax(zmax), .dbg_v(nn_dv), .dbg_d(nn_dd));

    wire        dp_dv;  wire [31:0] dp_dd;
    wk_dp u_dp (.clk(clk), .rst_n(rst_n), .clr(clr), .start(nn_done), .busy(dp_busy), .done(dp_done), .fire(dp_fire),
        .tpl_addr(tpl_addr), .sens(sens), .cfg_req(cfg_req), .cfg_addr(cfg_addr), .cfg_gnt(cfg_gnt), .cfg_rdata(cfg_rdata),
        .z_idx(z_idx), .z_val(z_val), .zmax(zmax), .dbg_v(dp_dv), .dbg_d(dp_dd));

    assign busy  = fe_busy | nn_busy | dp_busy | fe_start;
    assign dbg_v = fe_dv | nn_dv | dp_dv;
    assign dbg_k = fe_dv ? 2'd0 : nn_dv ? 2'd1 : 2'd2;
    assign dbg_d = fe_dv ? {16'd0, fe_dd} : nn_dv ? {10'd0, nn_dd} : dp_dd;
endmodule
