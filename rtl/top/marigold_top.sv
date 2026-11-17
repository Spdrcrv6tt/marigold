// Marigold chip top level (docs/architecture.md): PDM microphone in; I2C target, UART, irq and 8 GPIO out.
// Two clocks: clk_fe (3.072 MHz, the microphone's PDM clock) runs the always-on island: PDM front end, VAD, wake
// spotter, capture ring (mg_capture), and island control (mg_ctl: config, matcher, events, frames, GPIO, I2C).
// clk (the core oscillator, 98.304 MHz) runs the core (mg_core + weight ROM), which is held in reset (switched off)
// whenever core_on is low: core_on rises when a segment opens at a wake and falls after the core's power-off request
// with nothing queued. Crossings: samples from the ring as bundled data with a toggle (three-flop synchroniser, as
// Poinsetta); the island bridge as a bundled request {we, addr, wdata} with a toggle and a bundled answer with a toggle
// back. The island side of every core toggle is reset while the core is off, so both sides restart from zero.
module marigold_top #(
    parameter LANES     = 16,
    parameter FAN_L     = 16,
    parameter RAM_BYTES = 1 << 18,
    parameter PMEM_BYTES = 32768,
    parameter ROM_BYTES = 1 << 22,
    parameter ROM_INIT  = "",
    parameter CAP_AW    = 15,
    parameter [31:0] SEG_PRE = 32'd4800,
    parameter [31:0] SEG_MIN = 32'd8000,
    parameter [31:0] SEG_CAP = 32'd96000,
    parameter [7:0]  SEG_HANG = 8'd8,
    parameter [7:0]  SEG_ONSET = 8'd3,
    parameter [31:0] SEG_LISTEN = 32'd64000,
    parameter        MS = 3072,
    parameter        WAKE_STUB = 0,                     // 1: mg_wake_stub (wakes at tst_wake_at) instead of the spotter
    parameter        WAKE_MEMDIR = "",                 // wake spotter v2 ROM dir (rtl/wake/mem2/; chip.mk passes it)
    parameter        WAKE_LANES = 16,
    parameter        WAKE_C = 240,
    parameter        WAKE_NOUT = 48,
    parameter        WAKE_GATE = 1,
    parameter        WAKE_PREROLL = 10,
    parameter [15:0] MODEL_D = 16'd128   // the ROM model's D: a blob with another D is refused (review R4)
) (
    input  wire        clk,
    input  wire        clk_fe,
    input  wire        rst_n,
    output wire        pdm_clk_o,
    input  wire        pdm_dat_i,
    input  wire        scl_i,
    input  wire        sda_i,
    output wire        sda_oe,
    output wire        uart_tx_o,
    output wire        irq_o,
    output wire [7:0]  gpio_o,
    output wire        core_on,
    // test and debug (not bonded in a product)
    input  wire        tst_wake_en,
    input  wire [31:0] tst_wake_at,
    input  wire        trace_en_i,
    output wire        trace_valid,
    output wire [31:0] trace_tag,
    output wire [31:0] trace_val,
    output wire        halted,
    output wire        listening_o,
    output wire        fe_valid_o,
    output wire [15:0] fe_pcm_o,
    output wire        fe_raw_o,
    output wire        dbg_br_v,                        // a core write over the island bridge (clk_fe domain)
    output wire [15:0] dbg_br_a,
    output wire [31:0] dbg_br_d
);
    // ------------------------------------------------------------------ island (clk_fe)
    reg  [1:0] rfe;                                    // reset: asynchronous assert, synchronous release
    always @(posedge clk_fe or negedge rst_n) if (!rst_n) rfe <= 2'd0; else rfe <= {rfe[0], 1'b1};
    wire       rst_fe_n = rfe[1];

    wire               listening, fe_valid, fe_vad, fe_raw;
    wire signed [15:0] fe_data;
    pdm_frontend #(.DIV(1)) u_pdm (
        .clk(clk_fe), .rst_n(rst_fe_n), .en(listening), .gain(3'd0), .dc_en(1'b1),
        .pdm_clk_o(pdm_clk_o), .pdm_dat_i(pdm_dat_i), .pcm_valid(fe_valid), .pcm_data(fe_data), .overrun());
    vad u_vad (.clk(clk_fe), .rst_n(rst_fe_n), .in_valid(fe_valid), .in_data(fe_data), .vad_o(fe_vad),
               .frame_stb(), .raw_o(fe_raw), .e_o(), .z_o(), .nf_o(), .zn_o(), .nm_o(), .nd_o());
    assign listening_o = listening;
    assign fe_valid_o = fe_valid;
    assign fe_raw_o = fe_raw;
    assign fe_pcm_o = fe_data;

    wire [14:0] wake_base, spot_addr;
    wire [7:0]  wake_len;
    wire [15:0] wake_sens;
    wire        spot_req, spot_gnt, spot_rvalid, wake;
    wire [31:0] cfg_rdata;
    generate if (WAKE_STUB != 0) begin : g_stub
        mg_wake_stub u_wake (.clk(clk_fe), .rst_n(rst_fe_n), .en(listening), .in_valid(fe_valid), .in_data(fe_data),
            .in_vad(fe_vad), .in_raw(fe_raw), .tpl_addr(wake_base), .tpl_len(wake_len), .sens(wake_sens),
            .cfg_req(spot_req), .cfg_addr(spot_addr), .cfg_gnt(spot_gnt), .cfg_rdata(cfg_rdata), .wake(wake), .busy(),
            .tst_en(tst_wake_en), .tst_at(tst_wake_at));
    end else begin : g_wake                             // the wake agent's spotter (rtl/wake/)
        mg_wake #(.LANES(WAKE_LANES), .C(WAKE_C), .NOUT(WAKE_NOUT), .GATE(WAKE_GATE), .PREROLL(WAKE_PREROLL),
                  .MEMDIR(WAKE_MEMDIR), .ROM_STALL(0)) u_wake (.clk(clk_fe), .rst_n(rst_fe_n), .en(listening), .in_valid(fe_valid),
            .in_data(fe_data), .in_vad(fe_vad), .in_raw(fe_raw), .tpl_addr(wake_base), .tpl_len(wake_len),
            .sens(wake_sens), .cfg_req(spot_req), .cfg_addr(spot_addr), .cfg_gnt(spot_gnt), .cfg_rdata(cfg_rdata),
            .wake(wake), .busy(), .dbg_v(), .dbg_k(), .dbg_d());
    end endgenerate

    wire [31:0] q_idx, q_bkl;
    wire [15:0] q_data;
    wire        q_seg, q_tog, wk_tog, ack_tog, dq_tog, ev_wake, ev_tmo;
    mg_capture #(.AW(CAP_AW)) u_cap (
        .clk(clk_fe), .rst_n(rst_fe_n), .in_valid(fe_valid), .in_data(fe_data), .in_raw(fe_raw),
        .in_wake(wake && listening), .en(listening), .pre(SEG_PRE), .mn(SEG_MIN), .cap(SEG_CAP), .hang(SEG_HANG), .onset(SEG_ONSET),
        .listen(SEG_LISTEN), .ack_t(ack_tog), .dq_t(dq_tog), .q_idx(q_idx), .q_data(q_data), .q_seg(q_seg),
        .q_bkl(q_bkl), .q_tog(q_tog), .wk_t(wk_tog), .osc_en(core_on), .ev_wake(ev_wake), .ev_tmo(ev_tmo), .bank_pe());

    // island side of the bridge (reset while the core is off)
    wire        br_tog, br_we, i_ack;
    wire [15:0] br_addr;
    wire [31:0] br_wdata, i_rdata;
    reg  [2:0]  brs;
    reg         br_ack_t;
    reg  [31:0] br_rdata;
    wire        brst_n = rst_fe_n & core_on;
    always @(posedge clk_fe or negedge brst_n)
        if (!brst_n) begin brs <= 3'd0; br_ack_t <= 1'b0; br_rdata <= 32'd0; end
        else begin
            brs <= {brs[1:0], br_tog};
            if (i_ack) begin br_rdata <= i_rdata; br_ack_t <= ~br_ack_t; end
        end
    wire        i_req = brs[2] ^ brs[1];
    assign dbg_br_v = i_req && br_we;
    assign dbg_br_a = br_addr;
    assign dbg_br_d = br_wdata;

    mg_ctl #(.SEG_PRE(SEG_PRE), .SEG_MIN(SEG_MIN), .SEG_CAP(SEG_CAP), .SEG_HANG(SEG_HANG), .SEG_ONSET(SEG_ONSET),
             .SEG_LISTEN(SEG_LISTEN), .MS(MS), .MODEL_D(MODEL_D)) u_ctl (
        .clk(clk_fe), .rst_n(rst_fe_n), .scl_i(scl_i), .sda_i(sda_i), .sda_oe(sda_oe), .uart_tx_o(uart_tx_o),
        .irq_o(irq_o), .gpio_o(gpio_o), .ev_wake(ev_wake), .ev_tmo(ev_tmo), .cfg_valid(), .listening(listening),
        .wake_base(wake_base), .wake_len(wake_len), .wake_sens(wake_sens), .spot_req(spot_req), .spot_addr(spot_addr),
        .spot_gnt(spot_gnt), .spot_rvalid(spot_rvalid), .cfg_rdata(cfg_rdata),
        .br_req(i_req), .br_we(br_we), .br_addr(br_addr), .br_wdata(br_wdata), .br_ack(i_ack), .br_rdata(i_rdata));

    // ------------------------------------------------------------------ core (clk), off unless core_on
    // core reset: asserted at once when core_on falls (the domain is switched off and loses its state; the core
    // clock may already be stopped), released two core clocks after core_on rises with the clock running
    reg  [1:0]  con_s;
    wire        con_rst_n = rst_n & core_on;
    always @(posedge clk or negedge con_rst_n) if (!con_rst_n) con_s <= 2'd0; else con_s <= {con_s[0], 1'b1};
    wire        rst_c_n = con_s[1];
    reg  [2:0]  ts, wks;
    always @(posedge clk or negedge rst_c_n)
        if (!rst_c_n) begin ts <= 3'd0; wks <= 3'd0; end
        else begin ts <= {ts[1:0], q_tog}; wks <= {wks[1:0], wk_tog}; end
    wire        pcm_valid = ts[2] ^ ts[1];
    wire        wk_valid = wks[2] ^ wks[1];

    wire        seek_valid;
    wire [31:0] seek_addr;
    wire [8*LANES-1:0] ws_data;
    wire [5:0]  ws_avail, ws_take;
    imh_wsrc_rom #(.BYTES(ROM_BYTES), .WB(LANES), .INIT(ROM_INIT)) u_rom (
        .clk(clk), .rst_n(rst_c_n), .seek_valid(seek_valid), .seek_addr(seek_addr),
        .data(ws_data), .avail(ws_avail), .take(ws_take));

    mg_core #(.LANES(LANES), .FAN_L(FAN_L), .RAM_BYTES(RAM_BYTES), .PMEM_BYTES(PMEM_BYTES)) u_core (
        .clk(clk), .rst_n(rst_c_n),
        .ws_seek_valid(seek_valid), .ws_seek_addr(seek_addr), .ws_data(ws_data), .ws_avail(ws_avail), .ws_take(ws_take),
        .fl_mode(), .fl_dummy(), .fl_rx_dly(),
        .pcm_valid(pcm_valid), .pcm_data(q_data), .pcm_seg(q_seg), .pcm_idx_i(q_idx), .pcm_bkl_i(q_bkl),
        .pcm_ack_t(ack_tog), .dq_t(dq_tog), .wk_valid(wk_valid),
        .br_tog(br_tog), .br_we(br_we), .br_addr(br_addr), .br_wdata(br_wdata), .br_ack_t(br_ack_t), .br_rdata(br_rdata),
        .trace_en_i(trace_en_i), .trace_valid(trace_valid), .trace_tag(trace_tag), .trace_val(trace_val),
        .halted(halted));
endmodule
