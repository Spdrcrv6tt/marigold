// Imhotep compute core: controller + program memory + activation RAM + MAC array + registers.
// The weight source (on-die ROM or external flash) is outside, behind the seek / stream port, so the same
// core serves both builds. Register map: rtl/core/fw/hal.h. Description: docs/architecture.md.
module imh_core #(
    parameter LANES      = 16,
    parameter RAM_BYTES  = 1 << 20,
    parameter PMEM_BYTES = 32768,
    parameter UART_DIV   = 426,          // 49.152 MHz / 115200 - 1
    parameter [2:0] FL_MODE  = 3'd2,     // reset value of the flash read mode (quad output, 0x6C)
    parameter [4:0] FL_DUMMY = 5'd8
) (
    input  wire        clk,
    input  wire        rst_n,
    // weight source
    output wire        ws_seek_valid,
    output wire [31:0] ws_seek_addr,
    input  wire [8*LANES-1:0] ws_data,      // next bytes of the stream, low byte first
    input  wire [5:0]  ws_avail,              // how many of them are valid (0 = none yet)
    output wire [5:0]  ws_take,               // how many are consumed this clock
    output reg  [2:0]  fl_mode,
    output reg  [4:0]  fl_dummy,
    output reg  [1:0]  fl_rx_dly,
    // audio
    input  wire        pcm_valid,             // a sample from the capture buffer (rtl/top/imh_capture.sv)
    input  wire signed [15:0] pcm_data,
    input  wire        pcm_seg,               // with pcm_valid: 1 = a sample of the segment, 0 = the end marker
    input  wire [31:0] pcm_idx_i,             // with pcm_valid: its stream index (end marker: buffer high-water mark)
    input  wire [31:0] pcm_bkl_i,             // with pcm_valid: capture backlog when presented (REG_PCM_BKL)
    output reg         pcm_ack_t,             // toggles when the controller has taken the sample
    output reg  [31:0] pcm_pre,               // pre-roll and window cap of VAD-gated segments, samples (REG_PCM_PRE/CAP)
    output reg  [31:0] pcm_cap,
    output reg  [31:0] pcm_min,               // VAD-gated: no pause close before a segment holds this many samples (REG_PCM_MIN)
    output reg         dq_t,                  // toggles when the controller enters deep sleep (oscillator may stop)
    output reg         rexp,                  // deep sleep: the rec level the controller saw (wake when it differs)
    input  wire        wk_valid,              // the capture buffer's wake toggle, synchronised
    output reg         pdm_en,
    output reg  [2:0]  pdm_gain,
    output reg         pdm_dc_en,
    // pins
    output wire        uart_tx_o,
    input  wire        rec_i,
    input  wire        vadgate_i,             // high: recording is gated by the VAD (pin strap)
    input  wire        vad_i,
    input  wire        trace_en_i,
    output reg         busy_o,
    // trace / debug port
    output reg         trace_valid,
    output reg  [31:0] trace_tag,
    output reg  [31:0] trace_val,
    output wire        halted
);
    localparam W  = LANES * 16;
    localparam NB = LANES * 2;
    localparam LB = $clog2(LANES);

    // ------------------------------------------------------------------ core clock gate
    // Everything below runs on gclk except the always-on sample register and the sleep logic at the end of this
    // module. Writing REG_SLEEP stops gclk until a wake condition (a sample arrived; a sample with the VAD flag;
    // the rec pin changed), so while the controller waits for audio the core clock is stopped.
    wire gclk, core_en;
    clk_gate u_cg (.clk(clk), .en(core_en), .gclk(gclk));

    // ------------------------------------------------------------------ boot: copy the program out of the weight source
    localparam [1:0] B_SEEK = 2'd0, B_LEN = 2'd1, B_COPY = 2'd2, B_DONE = 2'd3;
    reg [1:0]  bst;
    reg [31:0] blen, bcnt;
    reg [1:0]  bl;
    wire       booting  = (bst != B_DONE);
    wire       ws_valid = (ws_avail != 6'd0);
    wire       boot_pop = ws_valid && (bst == B_LEN || bst == B_COPY);
    always @(posedge gclk or negedge rst_n) begin
        if (!rst_n) begin
            bst <= B_SEEK; blen <= 32'd0; bcnt <= 32'd0; bl <= 2'd0;
        end else begin
            case (bst)
                B_SEEK: bst <= B_LEN;
                B_LEN: if (ws_valid) begin
                    blen <= {ws_data[7:0], blen[31:8]};
                    bl   <= bl + 2'd1;
                    if (bl == 2'd3) bst <= B_COPY;
                end
                B_COPY: begin
                    if (bcnt == blen) bst <= B_DONE;
                    else if (ws_valid) bcnt <= bcnt + 32'd1;
                end
                default: ;
            endcase
        end
    end
    wire boot_we = (bst == B_COPY) && ws_valid && (bcnt != blen);

    // ------------------------------------------------------------------ controller
    wire        req, ready;
    wire [31:0] addr, wdata;
    wire [3:0]  wstrb;
    reg  [31:0] rdata;
    imh_cpu u_cpu (.clk(gclk), .rst_n(rst_n), .run(!booting), .req(req), .addr(addr), .wdata(wdata),
                   .wstrb(wstrb), .rdata(rdata), .ready(ready), .halted(halted));

    wire sel_pmem = (addr[31:28] == 4'h0);
    wire sel_ram  = (addr[31:28] == 4'h1);
    wire sel_mmio = (addr[31:28] == 4'h2);

    wire [31:0] pm_rdata;
    imh_pmem #(.BYTES(PMEM_BYTES)) u_pmem (.clk(gclk), .en(req && sel_pmem), .addr(addr), .rdata(pm_rdata),
                                            .bwe(boot_we), .baddr(bcnt), .bdata(ws_data[7:0]));

    // ------------------------------------------------------------------ activation RAM (shared with the MAC array)
    wire          mac_busy, mac_done, vec_busy, vec_done, fan_busy, fan_done;
    wire          ma_en, ve_en, fa_en;
    wire [NB-1:0] ma_we, ve_we, fa_we;
    wire [31:0]   ma_addr, ve_addr, fa_addr;
    wire [W-1:0]  ma_wdata, ve_wdata, fa_wdata, ram_rdata;
    wire          unit_busy = mac_busy | vec_busy | fan_busy;
    wire          mm_en    = mac_busy ? ma_en : vec_busy ? ve_en : fa_en;
    wire [NB-1:0] mm_we    = mac_busy ? ma_we : vec_busy ? ve_we : fa_we;
    wire [31:0]   mm_addr  = mac_busy ? ma_addr : vec_busy ? ve_addr : fa_addr;
    wire [W-1:0]  mm_wdata = mac_busy ? ma_wdata : vec_busy ? ve_wdata : fa_wdata;
    wire [LB:0]   bo = {addr[LB:2], 2'b00};                    // byte offset of the 32-bit access inside the word
    reg  [LB:0]   bo_r;
    wire          c_en = req && sel_ram;
    wire [NB-1:0] c_we = {{(NB-4){1'b0}}, wstrb} << bo;
    wire          ram_en    = unit_busy ? mm_en : c_en;
    wire [NB-1:0] ram_we    = unit_busy ? mm_we : c_we;
    wire [31:0]   ram_addr  = unit_busy ? mm_addr : {{(LB+5){1'b0}}, addr[27:LB+1]};
    wire [W-1:0]  ram_wdata = unit_busy ? mm_wdata : {(LANES/2){wdata}};
    imh_ram #(.LANES(LANES), .BYTES(RAM_BYTES)) u_ram (.clk(gclk), .en(ram_en), .we(ram_we), .addr(ram_addr),
                                                        .wdata(ram_wdata), .rdata(ram_rdata));

    // ------------------------------------------------------------------ MMIO
    reg         mp, mp_started;
    reg  [11:0] ma;
    reg  [31:0] md;
    reg         mw;
    reg  [1:0]  resp;             // 1 = pmem, 2 = ram answered this clock
    reg  [15:0] uart_div;
    reg  [15:0] pcm_d;              // always-on sample register (see the end of the module)
    reg         pcm_v, pcm_ovr, pcm_f;  // pcm_f: the segment flag of the sample in pcm_d
    reg  [31:0] pcm_idx, pcm_ridx;  // index of the sample in pcm_d; of the last one read
    reg  [31:0] pcm_bkl, pcm_rbkl;  // capture backlog with the sample in pcm_d; with the last one read
    reg         vg_s1, vg_s2, rec_p;
    reg         sleeping, woke, dpend;
    reg  [3:0]  wmask;
    reg  [31:0] ttag, cycles;
    reg         rec_s1, rec_s2;
    wire        uart_ready;
    wire [31:0] mac_argidx, mac_argval, fan_argidx, fan_argval;
    wire [255:0] fan_argall;                     // ARGMAX index of frames 0..7 (0x340 + 4c)
    wire        mac_sat;

    wire is_uart   = (ma == 12'h000), is_fldata = (ma == 12'h018), is_flseek = (ma == 12'h014);
    wire is_macreg = (ma[11:8] == 4'h1) && (ma[7:2] < 6'd12 || (ma[7:2] >= 6'd16 && ma[7:2] <= 6'd19));
    wire is_macgo  = (ma == 12'h130);
    wire is_vecreg = (ma[11:8] == 4'h2) && (ma[7:2] < 6'd13);
    wire is_vecgo  = (ma == 12'h234);
    wire is_fanreg = (ma[11:8] == 4'h3) && (ma[7:2] < 6'd9);
    wire is_fango  = (ma == 12'h330);
    wire is_sleep  = (ma == 12'h034);
    wire [63:0] vec_s, vec_q;
    wire [31:0] vec_mean, vec_rec, vec_sh1;
    wire mdone = mp && (is_uart && mw ? uart_ready :
                        is_fldata && !mw ? ws_valid :
                        is_macgo && mw ? mac_done :
                        is_vecgo && mw ? vec_done :
                        is_fango && mw ? fan_done :
                        is_sleep && mw ? woke : 1'b1);
    wire uart_push = mp && is_uart && mw && uart_ready;
    wire cpu_pop   = mp && is_fldata && !mw && ws_valid;
    wire cpu_seek  = mp && is_flseek && mw;
    wire mac_go    = mp && is_macgo && mw && !mp_started;
    wire vec_go    = mp && is_vecgo && mw && !mp_started;
    wire fan_go    = mp && is_fango && mw && !mp_started;
    reg  [31:0] mrd;
    always @* begin
        case (ma)
            12'h004: mrd = {16'd0, uart_div};
            12'h008: mrd = {27'd0, pdm_dc_en, pdm_gain, pdm_en};
            12'h00c: mrd = {13'd0, pcm_f, pcm_ovr, pcm_v, pcm_d};
            12'h010: mrd = {28'd0, vg_s2, trace_en_i, vad_i, rec_s2};
            12'h018: mrd = {24'd0, ws_data[7:0]};
            12'h024: mrd = RAM_BYTES;
            12'h028: mrd = cycles;
            12'h030: mrd = pcm_ridx;
            12'h040: mrd = pcm_rbkl;
            12'h044: mrd = pcm_min;
            12'h038: mrd = pcm_pre;
            12'h03c: mrd = pcm_cap;
            12'h02c: mrd = {22'd0, fl_rx_dly, fl_dummy, fl_mode};
            12'h134: mrd = mac_argidx;
            12'h138: mrd = mac_argval;
            12'h13c: mrd = {31'd0, mac_sat};
            12'h238: mrd = vec_s[31:0];
            12'h23c: mrd = vec_s[63:32];
            12'h240: mrd = vec_q[31:0];
            12'h244: mrd = vec_q[63:32];
            12'h248: mrd = vec_mean;
            12'h24c: mrd = vec_rec;
            12'h250: mrd = vec_sh1;
            12'h334: mrd = fan_argidx;
            12'h338: mrd = fan_argval;
            12'h340, 12'h344, 12'h348, 12'h34c, 12'h350, 12'h354, 12'h358, 12'h35c: mrd = fan_argall[32 * ma[4:2] +: 32];
            default: mrd = 32'd0;
        endcase
    end
    assign ready = (resp != 2'd0) || mdone;
    always @* begin
        case (resp)
            2'd1: rdata = pm_rdata;
            2'd2: rdata = ram_rdata[8 * bo_r +: 32];
            default: rdata = mrd;
        endcase
    end

    always @(posedge gclk or negedge rst_n) begin
        if (!rst_n) begin
            mp <= 1'b0; mp_started <= 1'b0; ma <= 12'd0; md <= 32'd0; mw <= 1'b0; resp <= 2'd0; bo_r <= 0;
            uart_div <= UART_DIV; ttag <= 32'd0; cycles <= 32'd0;
            pdm_en <= 1'b0; pdm_gain <= 3'd0; pdm_dc_en <= 1'b1; busy_o <= 1'b0;
            fl_mode <= FL_MODE; fl_dummy <= FL_DUMMY; fl_rx_dly <= 2'd0; pcm_pre <= 32'd0; pcm_cap <= 32'd0; pcm_min <= 32'd0;
        end else begin
            cycles <= cycles + 32'd1;
            resp <= (req && sel_pmem) ? 2'd1 : (req && sel_ram) ? 2'd2 : 2'd0;
            if (c_en) bo_r <= bo;
            if (req && sel_mmio) begin
                mp <= 1'b1; mp_started <= 1'b0; ma <= addr[11:0]; md <= wdata; mw <= (wstrb != 4'd0);
            end else if (mp) begin
                if (mac_go || vec_go || fan_go) mp_started <= 1'b1;
                if (mdone) begin
                    mp <= 1'b0;
                    if (mw) begin
                        case (ma)
                            12'h004: uart_div <= md[15:0];
                            12'h008: begin pdm_en <= md[0]; pdm_gain <= md[3:1]; pdm_dc_en <= md[4]; end
                            12'h010: busy_o <= md[0];
                            12'h01c: ttag <= md;
                            12'h02c: begin fl_mode <= md[2:0]; fl_dummy <= md[7:3]; fl_rx_dly <= md[9:8]; end
                            12'h038: pcm_pre <= md;
                            12'h03c: pcm_cap <= md;
                            12'h044: pcm_min <= md;
                            default: ;
                        endcase
                    end
                end
            end
        end
    end

    uart_tx #(.DW(16)) u_uart (.clk(gclk), .rst_n(rst_n), .div(uart_div), .in_data(md[7:0]),
                               .in_valid(uart_push), .in_ready(uart_ready), .tx_o(uart_tx_o));

    wire        mac_seek, vec_seek, fan_seek;
    wire [31:0] mac_seek_addr, vec_seek_addr, fan_seek_addr;
    wire [5:0]  mac_take, vec_take, fan_take;
    // fan-out array (Poinsetta): 32 output rows per group, up to 8 frames per weight read (imh_fan.sv)
    imh_fan #(.L(32), .CMAX(8), .RWL(LANES), .WB(LANES)) u_fan (
        .clk(gclk), .rst_n(rst_n),
        .reg_we(mp && mw && is_fanreg), .reg_idx(ma[6:2]), .reg_wdata(md), .go(fan_go), .busy(fan_busy), .done(fan_done),
        .argidx(fan_argidx), .argval(fan_argval), .argidx_all(fan_argall),
        .m_en(fa_en), .m_we(fa_we), .m_addr(fa_addr), .m_wdata(fa_wdata), .m_rdata(ram_rdata),
        .seek_valid(fan_seek), .seek_addr(fan_seek_addr), .ws_data(ws_data), .ws_avail(fan_busy ? ws_avail : 6'd0),
        .ws_take(fan_take));
    imh_mac #(.LANES(LANES)) u_mac (
        .clk(gclk), .rst_n(rst_n),
        .reg_we(mp && mw && is_macreg), .reg_idx(ma[6:2]), .reg_wdata(md), .go(mac_go), .busy(mac_busy), .done(mac_done),
        .argidx(mac_argidx), .argval(mac_argval), .satflag(mac_sat),
        .m_en(ma_en), .m_we(ma_we), .m_addr(ma_addr), .m_wdata(ma_wdata), .m_rdata(ram_rdata),
        .seek_valid(mac_seek), .seek_addr(mac_seek_addr), .ws_data(ws_data), .ws_avail(mac_busy ? ws_avail : 6'd0), .ws_take(mac_take));

    imh_vec #(.LANES(LANES)) u_vec (
        .clk(gclk), .rst_n(rst_n),
        .reg_we(mp && mw && is_vecreg), .reg_idx(ma[6:2]), .reg_wdata(md), .go(vec_go), .busy(vec_busy), .done(vec_done),
        .s_acc(vec_s), .q_acc(vec_q), .o_mean(vec_mean), .o_rec(vec_rec), .o_sh1(vec_sh1),
        .m_en(ve_en), .m_we(ve_we), .m_addr(ve_addr), .m_wdata(ve_wdata), .m_rdata(ram_rdata),
        .seek_valid(vec_seek), .seek_addr(vec_seek_addr), .ws_data(ws_data), .ws_avail(vec_busy ? ws_avail : 6'd0), .ws_take(vec_take));

    // ------------------------------------------------------------------ always-on part (free-running clk)
    // Sample register (filled from the front-end clock domain, see imhotep_top), pin synchronisers, the trace-port
    // strobe and the sleep / wake logic. The gated side talks to it with one-clock events that only occur while
    // gclk runs. REG_SLEEP write data: [0] wake on a new sample, [1] on a sample of a segment, [2] on a rec change;
    // [3] deep sleep: once the UART is idle the capture buffer is told (dq_t) and may stop the core oscillator; the
    // only wake is then its wake toggle (a segment is ready, or rec differs from [4], the level the controller saw).
    wire pcm_rd   = mp && mdone && !mw && (ma == 12'h00c);
    wire tr_wr    = mp && mdone && mw && (ma == 12'h020);
    wire sleep_rq = mp && mw && is_sleep;
    wire wake     = wmask[3] ? wk_valid :
                    (wmask[0] && pcm_v) || (wmask[1] && pcm_v && pcm_f) || (wmask[2] && (rec_s2 != rec_p));
    wire dq_go    = dpend && uart_ready;
    assign core_en = !sleeping || !uart_ready;
    // the always-on registers get a clock edge only when something happens to them
    wire ao_en = pcm_valid || pcm_rd || tr_wr || trace_valid || (sleep_rq && !sleeping && !woke) || (sleeping && wake) || woke
                 || dq_go;
    wire aclk;
    clk_gate u_cga (.clk(clk), .en(ao_en), .gclk(aclk));
    always @(posedge clk or negedge rst_n)                // rec pin synchroniser: free-running (wakes the core)
        if (!rst_n) begin rec_s1 <= 1'b0; rec_s2 <= 1'b0; end
        else begin rec_s1 <= rec_i; rec_s2 <= rec_s1; end
    always @(posedge gclk or negedge rst_n)               // strap synchroniser
        if (!rst_n) begin vg_s1 <= 1'b0; vg_s2 <= 1'b0; end
        else begin vg_s1 <= vadgate_i; vg_s2 <= vg_s1; end
    always @(posedge aclk or negedge rst_n) begin
        if (!rst_n) begin
            pcm_d <= 16'd0; pcm_v <= 1'b0; pcm_ovr <= 1'b0; pcm_f <= 1'b0; pcm_idx <= 32'd0; pcm_ridx <= 32'd0;
            pcm_bkl <= 32'd0; pcm_rbkl <= 32'd0;
            pcm_ack_t <= 1'b0; dq_t <= 1'b0; rexp <= 1'b0; dpend <= 1'b0;
            rec_p <= 1'b0; sleeping <= 1'b0; woke <= 1'b0; wmask <= 4'd0;
            trace_valid <= 1'b0; trace_tag <= 32'd0; trace_val <= 32'd0;
        end else begin
            trace_valid <= tr_wr;
            if (tr_wr) begin trace_tag <= ttag; trace_val <= md; end
            if (pcm_valid) begin
                pcm_d <= pcm_data; pcm_v <= 1'b1; pcm_f <= pcm_seg; pcm_idx <= pcm_idx_i; pcm_bkl <= pcm_bkl_i;
                if (pcm_v) pcm_ovr <= 1'b1;
            end else if (pcm_rd && pcm_v) begin pcm_v <= 1'b0; pcm_ack_t <= ~pcm_ack_t; end   // taken: ask for the next
            if (pcm_rd) begin pcm_ridx <= pcm_idx; pcm_rbkl <= pcm_bkl; end   // index and backlog of the sample this read returns
            if (!sleeping && !woke && sleep_rq) begin
                sleeping <= 1'b1; wmask <= md[3:0]; rec_p <= rec_s2;
                if (md[3]) begin dpend <= 1'b1; rexp <= md[4]; end
            end
            else if (sleeping && wake) begin sleeping <= 1'b0; woke <= 1'b1; end
            else if (woke && !sleep_rq) woke <= 1'b0;
            if (dq_go) begin dpend <= 1'b0; dq_t <= ~dq_t; end
        end
    end

    // weight source port: boot, then the array, the per-element unit or the controller (one at a time: the
    // controller is stalled while a unit runs)
    assign ws_seek_valid = (bst == B_SEEK) || mac_seek || vec_seek || fan_seek || cpu_seek;
    assign ws_seek_addr  = (bst == B_SEEK) ? 32'd4 : mac_seek ? mac_seek_addr : vec_seek ? vec_seek_addr :
                           fan_seek ? fan_seek_addr : md;
    assign ws_take       = (boot_pop || cpu_pop) ? 6'd1 : mac_busy ? mac_take : vec_busy ? vec_take : fan_busy ? fan_take : 6'd0;
endmodule
