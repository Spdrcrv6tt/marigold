// Imhotep chip top level: PDM microphone in, text out on the UART.
//   WSRC_ROM = 1: hardwired build, the weights and the program are in an on-die ROM (the flash pins idle).
//   WSRC_ROM = 0: the same chip with the weights in an external NOR flash (quad or octal SPI).
// rec_i high = record; when it drops the utterance is transcribed and the text is sent, ending in a newline.
// vadgate_i high: rec_i only arms the chip; the VAD starts and ends each utterance (several per reset).
// Two clocks: clk_fe (3.072 MHz, the microphone's PDM clock) runs the PDM decimator, the VAD and the capture
// buffer (imh_capture: every sample is kept in a ring, segments are decided there with a pre-roll, and speech that
// arrives while the core transcribes waits there) and is the only clock running while listening; clk (49.152 MHz,
// or about 100 MHz at the dense-node design point) runs the core. Its oscillator is enabled from the clk_fe domain
// (osc_en_o): stopped while the core sleeps with nothing to do, started when a segment is ready or rec changes.
// Samples cross from clk_fe to clk as bundled data {index, segment flag, sample} held until the core answers,
// announced by a toggle synchronised into clk by three flops; the core answers with a toggle.
module imhotep_top #(
    parameter LANES      = 16,
    parameter RAM_BYTES  = 1 << 20,
    parameter PMEM_BYTES = 32768,
    parameter WSRC_ROM   = 1,
    parameter ROM_BYTES  = 1 << 25,
    parameter ROM_INIT   = "",
    parameter IOW        = 4,
    parameter UART_DIV   = 426,
    parameter CAP_AW     = 16          // capture buffer: 2^CAP_AW samples of 16 bits (128 KB: reports/p1-core.md section 15)
) (
    input  wire           clk,          // core clock, 49.152 MHz
    input  wire           clk_fe,       // front-end clock, 3.072 MHz
    input  wire           rst_n,
    input  wire           vadgate_i,
    output wire           pdm_clk_o,
    input  wire           pdm_dat_i,
    output wire           fl_sck_o,
    output wire           fl_cs_n_o,
    output wire [IOW-1:0] fl_io_o,
    output wire [IOW-1:0] fl_io_oe,
    input  wire [IOW-1:0] fl_io_i,
    output wire           uart_tx_o,
    input  wire           rec_i,
    output wire           busy_o,
    output wire           vad_o,
    input  wire           trace_en_i,
    output wire           trace_valid,
    output wire [31:0]    trace_tag,
    output wire [31:0]    trace_val,
    output wire           halted,
    output wire           osc_en_o      // core oscillator enable (clk runs only while it is high)
);
    wire        seek_valid;
    wire [31:0] seek_addr;
    wire [8*LANES-1:0] ws_data;
    wire [5:0]  ws_avail, ws_take;
    wire [2:0]  fl_mode;
    wire [4:0]  fl_dummy;
    wire [1:0]  fl_rx_dly;
    wire        pdm_en, pdm_dc_en;
    wire [2:0]  pdm_gain;

    // ---- front-end clock domain
    reg  [1:0]  rfe;                                   // reset: asynchronous assert, synchronous release
    always @(posedge clk_fe or negedge rst_n) if (!rst_n) rfe <= 2'd0; else rfe <= {rfe[0], 1'b1};
    wire        rst_fe_n = rfe[1];
    reg  [1:0]  en_s, dc_s;                            // controls from the core: quasi-static, two-flop synchronised
    reg  [2:0]  g_s0, g_s1;
    always @(posedge clk_fe or negedge rst_fe_n)
        if (!rst_fe_n) begin en_s <= 2'd0; dc_s <= 2'b11; g_s0 <= 3'd0; g_s1 <= 3'd0; end
        else begin en_s <= {en_s[0], pdm_en}; dc_s <= {dc_s[0], pdm_dc_en}; g_s0 <= pdm_gain; g_s1 <= g_s0; end
    wire               fe_valid, fe_vad;
    wire signed [15:0] fe_data;
    pdm_frontend #(.DIV(1)) u_pdm (
        .clk(clk_fe), .rst_n(rst_fe_n), .en(en_s[1]), .gain(g_s1), .dc_en(dc_s[1]),
        .pdm_clk_o(pdm_clk_o), .pdm_dat_i(pdm_dat_i), .pcm_valid(fe_valid), .pcm_data(fe_data), .overrun());
    vad u_vad (.clk(clk_fe), .rst_n(rst_fe_n), .in_valid(fe_valid), .in_data(fe_data), .vad_o(fe_vad),
               .frame_stb(), .raw_o(), .e_o(), .z_o(), .nf_o(), .zn_o());
    assign vad_o = fe_vad;
    wire [31:0] cap_pre, cap_max, cap_min, q_idx;
    wire [15:0] q_data;
    wire        q_seg, q_tog, wk_tog, ack_tog, dq_tog, rexp;
    wire [31:0] q_bkl;
    imh_capture #(.AW(CAP_AW)) u_cap (
        .clk(clk_fe), .rst_n(rst_fe_n), .in_valid(fe_valid), .in_data(fe_data), .in_vad(fe_vad),
        .rec_i(rec_i), .vg_i(vadgate_i), .pre(cap_pre), .cap(cap_max), .mn(cap_min), .rexp(rexp), .ack_t(ack_tog), .dq_t(dq_tog),
        .q_idx(q_idx), .q_data(q_data), .q_seg(q_seg), .q_bkl(q_bkl), .q_tog(q_tog), .wk_t(wk_tog), .osc_en(osc_en_o));

    // ---- into the core clock: the sample-toggle and wake-toggle synchronisers (with the rec pin synchroniser in
    // imh_core, the only core-clock flops clocked while the core sleeps with the oscillator running)
    reg  [2:0]  ts, wks;
    always @(posedge clk or negedge rst_n)
        if (!rst_n) begin ts <= 3'd0; wks <= 3'd0; end
        else begin ts <= {ts[1:0], q_tog}; wks <= {wks[1:0], wk_tog}; end
    wire        pcm_valid = ts[2] ^ ts[1];
    wire        wk_valid = wks[2] ^ wks[1];

    generate
        if (WSRC_ROM != 0) begin : g_rom
            imh_wsrc_rom #(.BYTES(ROM_BYTES), .WB(LANES), .INIT(ROM_INIT)) u_ws (
                .clk(clk), .rst_n(rst_n), .seek_valid(seek_valid), .seek_addr(seek_addr),
                .data(ws_data), .avail(ws_avail), .take(ws_take));
            assign fl_sck_o = 1'b0;
            assign fl_cs_n_o = 1'b1;
            assign fl_io_o = {IOW{1'b1}};
            assign fl_io_oe = {IOW{1'b0}};
        end else begin : g_flash
            imh_wsrc_flash #(.IOW(IOW), .WB(LANES)) u_ws (
                .clk(clk), .rst_n(rst_n), .mode(fl_mode), .dummy(fl_dummy), .rx_dly(fl_rx_dly),
                .seek_valid(seek_valid), .seek_addr(seek_addr),
                .data(ws_data), .avail(ws_avail), .take(ws_take),
                .sck_o(fl_sck_o), .cs_n_o(fl_cs_n_o), .io_o(fl_io_o), .io_oe(fl_io_oe), .io_i(fl_io_i));
        end
    endgenerate

    imh_core #(.LANES(LANES), .RAM_BYTES(RAM_BYTES), .PMEM_BYTES(PMEM_BYTES), .UART_DIV(UART_DIV)) u_core (
        .clk(clk), .rst_n(rst_n),
        .ws_seek_valid(seek_valid), .ws_seek_addr(seek_addr), .ws_data(ws_data), .ws_avail(ws_avail), .ws_take(ws_take),
        .fl_mode(fl_mode), .fl_dummy(fl_dummy), .fl_rx_dly(fl_rx_dly),
        .pcm_valid(pcm_valid), .pcm_data(q_data), .pcm_seg(q_seg), .pcm_idx_i(q_idx), .pcm_bkl_i(q_bkl), .pcm_ack_t(ack_tog),
        .pcm_pre(cap_pre), .pcm_cap(cap_max), .pcm_min(cap_min), .dq_t(dq_tog), .rexp(rexp), .wk_valid(wk_valid),
        .pdm_en(pdm_en), .pdm_gain(pdm_gain), .pdm_dc_en(pdm_dc_en),
        .uart_tx_o(uart_tx_o), .rec_i(rec_i), .vadgate_i(vadgate_i), .vad_i(fe_vad), .trace_en_i(trace_en_i), .busy_o(busy_o),
        .trace_valid(trace_valid), .trace_tag(trace_tag), .trace_val(trace_val), .halted(halted));
endmodule
