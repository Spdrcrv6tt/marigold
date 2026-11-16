// Marigold wake spotter (docs/golden-wake.md; interface docs/architecture.md section 3.1, rules agreed Nov 15).
// clk_fe domain (3.072 MHz). Sample n = 0 is the first in_valid with en = 1; while en = 0 all state clears.
// Real-time frame t' covers samples [320t', 320t'+560) and is triggered by sample 320t'+559 (ring of 1024 samples).
// Each trigger runs the front end for frame t' (features into a ring of PREROLL+1 slots). Format 2 (GATE = 1): the
// spotter samples in_vad at the trigger; the phone network and keyword DP then run frame t = t' - PREROLL only if
// in_vad was 1 at any of the triggers t .. t'; a run of such frames starts with zero network history, fresh DP
// states and refractory 0. A fire for frame t raises `wake` with the in_valid of sample k = 320(t + PREROLL) + 879.
// Format 1 (GATE = 0, PREROLL = 0): every frame runs, k = 320t + 879. Engines have their own clock gates.
// tpl_len and in_raw are not needed (the template is self-describing).
module mg_wake #(
    parameter MEMDIR    = "",
    parameter ROM_STALL = 0,
    parameter LANES     = 16,
    parameter C         = 240,
    parameter NOUT      = 48,
    parameter GATE      = 1,
    parameter PREROLL   = 10
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
    output wire [1:0]         dbg_k,         // 0 feature {frame b, band, v}, 1 logit {idx, z}, 2 best score
    output wire [31:0]        dbg_d
);
    localparam integer NS = PREROLL + 1;     // feature slots
    wire clr = !en;

    // ---- sample ring, frame schedule, VAD look-ahead
    reg  [9:0] wptr, left, fbase;
    reg        fire_q, ovr;
    reg  [3:0] fslot;                        // feature slot of the current real-time frame
    reg  [PREROLL:0] vsr;                    // in_vad at the last PREROLL+1 triggers
    reg  [4:0] nfr;                          // real-time frames seen, saturating at PREROLL
    reg        go, prev_go, go_fresh;
    wire       acc  = en & in_valid;
    wire       trig = acc & (left == 10'd1);
    assign wake = trig & fire_q;
    wire        sr_re;
    wire [9:0]  sr_addr;
    wire [15:0] sr_rdata;
    wk_ram #(.WORDS(1024), .WIDTH(16)) u_ring (.clk(clk), .we(acc), .waddr(wptr), .wdata(in_data), .re(sr_re), .raddr(sr_addr), .rdata(sr_rdata));

    wire fe_busy, fe_done, nn_busy, nn_done, dp_busy, dp_done, dp_fire;
    reg  fe_start;
    wire [PREROLL+1:0] vcat = {vsr, in_vad};
    wire [PREROLL:0]   vnext = vcat[PREROLL:0];
    wire go_now = (GATE == 0) ? 1'b1 : ((nfr >= 5'(PREROLL)) && (|vnext));
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wptr <= 0; left <= 10'd560; fire_q <= 0; fe_start <= 0; fbase <= 0; ovr <= 0; fslot <= 0; vsr <= 0; nfr <= 0;
            go <= 0; prev_go <= 0; go_fresh <= 0;
        end else if (clr) begin
            wptr <= 0; left <= 10'd560; fire_q <= 0; fe_start <= 0; fslot <= 0; vsr <= 0; nfr <= 0; go <= 0; prev_go <= 0;
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
                    vsr <= vnext;
                    nfr <= (nfr >= 5'(PREROLL)) ? nfr : nfr + 1;
                    fslot <= (fslot == 4'(NS - 1)) ? 4'd0 : fslot + 1;
                    go <= go_now;                         // run frame t' - PREROLL after this front end
                    go_fresh <= go_now & ~prev_go;
                    prev_go <= go_now;
                end
            end
            if (dp_fire) fire_q <= 1;
        end
    end
    // slots: the front end writes frame t' at fslot (as latched at the trigger: the value before the increment),
    // the network reads frame t' - PREROLL at the next slot
    reg [3:0] wslot;
    always @(posedge clk or negedge rst_n)
        if (!rst_n) wslot <= 0; else if (clr) wslot <= 0; else if (trig) wslot <= fslot;
    wire [3:0] rslot = (wslot == 4'(NS - 1)) ? 4'd0 : wslot + 1;

    // ---- per-engine clock gates
    wire nn_start = fe_done & go;
    wire g_fe, g_nn, g_dp;
    clk_gate u_cg_fe (.clk(clk), .en(fe_start | fe_busy | fe_done | clr), .gclk(g_fe));
    clk_gate u_cg_nn (.clk(clk), .en(nn_start | nn_busy | nn_done | clr), .gclk(g_nn));
    clk_gate u_cg_dp (.clk(clk), .en(nn_done | dp_busy | dp_done | clr), .gclk(g_dp));

    // ---- engines
    wire       x_we;  wire [5:0] x_addr;  wire [7:0] x_data;
    wire       fe_dv; wire [15:0] fe_dd;
    wk_fe #(.MEMDIR(MEMDIR)) u_fe (.clk(g_fe), .rst_n(rst_n), .clr(clr), .start(fe_start), .base(fbase), .busy(fe_busy), .done(fe_done),
        .sr_re(sr_re), .sr_addr(sr_addr), .sr_rdata(sr_rdata), .x_we(x_we), .x_addr(x_addr), .x_data(x_data), .dbg_v(fe_dv), .dbg_d(fe_dd));

    wire        rom_req, rom_gnt, rom_rvalid;
    wire [16:0] rom_addr;
    wire [4*LANES-1:0] rom_rdata;
    wk_rom #(.W(4*LANES), .STALL(ROM_STALL), .MEMFILE({MEMDIR, "wake_rom.mem"})) u_rom (.clk(g_nn), .rst_n(rst_n), .rom_req(rom_req),
        .rom_addr(rom_addr), .rom_gnt(rom_gnt), .rom_rvalid(rom_rvalid), .rom_rdata(rom_rdata));

    wire [5:0]  z_idx;  wire signed [15:0] z_val, zmax;
    wire        nn_dv;  wire [21:0] nn_dd;
    wk_nn #(.LANES(LANES), .C(C), .NOUT(NOUT), .XW(10)) u_nn (.clk(g_nn), .clk_x(clk), .rst_n(rst_n), .clr(clr), .start(nn_start),
        .fresh(go_fresh), .x_base({rslot, 6'd0}), .busy(nn_busy), .done(nn_done),
        .x_we(x_we), .x_addr({wslot, x_addr}), .x_data(x_data), .rom_req(rom_req), .rom_addr(rom_addr), .rom_gnt(rom_gnt),
        .rom_rvalid(rom_rvalid), .rom_rdata(rom_rdata), .z_idx(z_idx), .z_val(z_val), .zmax(zmax), .dbg_v(nn_dv), .dbg_d(nn_dd));

    wire        dp_dv;  wire [31:0] dp_dd;
    wk_dp u_dp (.clk(g_dp), .rst_n(rst_n), .clr(clr), .start(nn_done), .fresh(go_fresh), .busy(dp_busy), .done(dp_done), .fire(dp_fire),
        .tpl_addr(tpl_addr), .sens(sens), .cfg_req(cfg_req), .cfg_addr(cfg_addr), .cfg_gnt(cfg_gnt), .cfg_rdata(cfg_rdata),
        .z_idx(z_idx), .z_val(z_val), .zmax(zmax), .dbg_v(dp_dv), .dbg_d(dp_dd));

    assign busy  = fe_busy | nn_busy | dp_busy | fe_start;
    assign dbg_v = fe_dv | nn_dv | dp_dv;
    assign dbg_k = fe_dv ? 2'd0 : nn_dv ? 2'd1 : 2'd2;
    assign dbg_d = fe_dv ? {16'd0, fe_dd} : nn_dv ? {10'd0, nn_dd} : dp_dd;
endmodule
