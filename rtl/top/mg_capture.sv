// Capture ring, wake-gated segmenter and sample delivery (from Poinsetta's imh_capture.sv), front-end clock domain.
//
// Every sample from the PDM front end goes into a ring of 2^AW samples (Marigold: 2^15 = 64 KB in 16 banks of 4 KB)
// whatever the core is doing, so the pre-roll before a wake and the backlog while the core boots are kept.
// Segments follow docs/architecture.md section 8.1, decided live for the sample being written (index wp):
//   frame boundary fb: wp a positive multiple of 256; in_raw with that sample is frame wp/256 - 1's raw flag.
//   h  = consecutive frames with raw = 0 (from reset); sp = frames with raw = 1 that start at or after the wake.
//   closed + in_wake at wp: open at start = max(wp - pre, end of the previous segment); k = wp; WAKE event.
//   open: at fb, before speech (sp < onset) close at the first wp with wp - k >= listen (TIMEOUT); after speech close
//   at the first wp with wp - start >= mn and h >= hang; at any sample close when wp - start == cap (cap != 0).
// Segments queue (QD deep) and are delivered to the core in order, one sample per handshake: {index, sample,
// seg = 1} in registers that hold still until the core answers (bundled data), announced by a toggle (q_tog) and
// taken with a toggle back (ack_t). After a segment's last sample comes one end marker: seg = 0, index = the
// high-water mark of the buffer, data[0] = overrun (sticky), data[1] = this segment closed as a TIMEOUT.
// Core power: the core asks to sleep with a toggle (dq_t). Unless something is queued or being delivered, the core
// oscillator and power are switched off (osc_en low); when a segment opens they come back (wk_t toggles).
// Ring banks: 2^(AW-BW) banks of 2^BW samples. While no segment is open, queued or being delivered only the banks
// holding the last pre + 1 samples (the window a wake could start a segment from) are powered (bank_pe); all are
// powered while a segment is open. In simulation a bank powered down loses its contents (random fill).
module mg_capture #(
    parameter AW = 15,           // ring of 2^AW samples (2^AW x 16 bit)
    parameter BW = 11,           // banks of 2^BW samples (4 KB)
    parameter QD = 2,            // queued segments (power of two)
    parameter [31:0] WP0 = 32'd0 // reset value of the sample index (tests start it near the 2^32 wrap)
) (
    input  wire               clk,
    input  wire               rst_n,
    input  wire               in_valid,
    input  wire signed [15:0] in_data,
    input  wire               in_raw,      // raw flag bundled with this sample (frame wp/256 - 1)
    input  wire               in_wake,     // the wake spotter fired with this sample
    input  wire [31:0]        pre,
    input  wire [31:0]        mn,
    input  wire [31:0]        cap,
    input  wire [7:0]         hang,
    input  wire [7:0]         onset,
    input  wire [31:0]        listen,
    input  wire               ack_t,       // from the core clock domain (synchronised here)
    input  wire               dq_t,
    output reg  [31:0]        q_idx,
    output reg  [15:0]        q_data,
    output reg                q_seg,
    output reg  [31:0]        q_bkl,       // with a sample: backlog (written, not yet delivered) when presented
    output reg                q_tog,
    output reg                wk_t,
    output reg                osc_en,
    output reg                ev_wake,     // one clock: a segment opened (WAKE)
    output reg                ev_tmo,      // one clock: a segment closed as a TIMEOUT
    output wire [(1 << (AW - BW)) - 1:0] bank_pe
);
    localparam QB = $clog2(QD);
    localparam [31:0] R = 32'd1 << AW;

    // ---- free-running: synchronisers and the core power control
    reg  [1:0] ack_s, dq_s;
    reg        dq_seen, slp;
    reg  [2:0] offc;                                         // clocks the core must stay off
    wire       dq  = (dq_s[1] != dq_seen);
    reg        rd_act, outst, ack_seen;
    reg  [QB:0] qh, qt;
    wire       qempty = (qh == qt);
    wire       ev = rd_act || !qempty || outst;
    always @(posedge clk or negedge rst_n)
        if (!rst_n) begin
            ack_s <= 2'd0; dq_s <= 2'd0; dq_seen <= 1'b0; slp <= 1'b1; wk_t <= 1'b0; osc_en <= 1'b0; offc <= 3'd0;
        end else begin
            ack_s <= {ack_s[0], ack_t}; dq_s <= {dq_s[0], dq_t};
            if (!osc_en) begin                               // off: the core's toggles are at their reset value
                dq_seen <= 1'b0;
                if (offc != 3'd0) offc <= offc - 3'd1;
            end
            if (osc_en && dq) begin                          // power-off request: switch off, even with a segment
                osc_en <= 1'b0; slp <= 1'b1; offc <= 3'd4;   // waiting (it boots again for it: review R2)
            end else if (slp && ev && offc == 3'd0) begin
                osc_en <= 1'b1; slp <= 1'b0; wk_t <= ~wk_t;
            end
        end

    // ---- everything else runs on a gated clock: once per sample while listening, every clock while delivering
    wire ack = osc_en && (ack_s[1] != ack_seen);             // no acks while the core is off (review R8)

    reg  [15:0] mem [0:(1 << AW) - 1];
    reg  [15:0] rq;
    reg  [31:0] wp, last_end, sstart, kw, rp, hwm;
    reg         wfull, le_v;                                 // pre samples written since reset; last_end recent
    reg  [7:0]  h, sp;
    reg         on, oq, rpend, mark, ovr;
    reg  [31:0] qs [0:QD-1];
    reg  [31:0] qe [0:QD-1];
    reg  [QD-1:0] qc, qto;
    localparam [QB:0] QDV = QD;
    wire [QB-1:0] hi = qh[QB-1:0], ti = qt[QB-1:0], tl = ti - 1'b1;
    wire        qfull = ((qt - qh) == QDV);

    // segmenter, for the sample being written (index wp)
    wire [31:0] ws     = wp - WP0;                                 // samples since the stream started
    wire        fb     = (ws[7:0] == 8'd0) && (ws != 32'd0);
    wire [7:0]  h_n    = fb ? (in_raw ? 8'd0 : ((h == 8'hFF) ? h : h + 8'd1)) : h;
    // start = max(wp - pre, end of the previous segment, stream start), with modular differences only (review R1):
    // last_end counts only while it is within pre samples of wp (le_v drops once it is older)
    wire        full_n = wfull || (ws >= pre);
    wire [31:0] kp     = full_n ? (wp - pre) : WP0;
    wire [31:0] st0    = (le_v && ((wp - last_end) < pre)) ? last_end : kp;
    wire        wopen  = !on && in_wake;
    wire [31:0] dk     = wp - 32'd256 - kw;                       // frame start minus the wake, while open
    wire        spc    = fb && in_raw && !dk[31];
    wire [7:0]  sp_n   = (spc && sp != 8'hFF) ? sp + 8'd1 : sp;
    wire        c_tmo  = on && fb && (sp_n < onset) && (wp - kw >= listen);
    wire        c_sp   = on && fb && (sp_n >= onset) && (wp - sstart >= mn) && (h_n >= hang);
    wire        c_cap  = on && (cap != 32'd0) && ((wp - sstart) == cap);
    wire        wclose = c_tmo || c_sp || c_cap;
    // reader
    wire        mk     = osc_en && rd_act && !outst && !rpend && qc[hi] && (rp == qe[hi]);       // end marker due
    wire        re     = osc_en && rd_act && !outst && !rpend && !mk && (rp != wp) && !in_valid;  // read the next sample
    wire        rwork  = osc_en && rd_act && !outst && (rpend || (rp != wp) || (qc[hi] && (rp == qe[hi])));
    wire        rsync  = !osc_en && (q_tog || ack_seen || outst || rpend || mark);  // core off: restart the handshake
    // buffer use: unread samples of queued segments, counting the one being written
    wire        need   = rd_act || !qempty;
    wire [31:0] oldest = rd_act ? rp : qs[hi];
    wire [31:0] occ_n  = need ? (wp + 32'd1 - oldest) : 32'd0;
    wire [31:0] occ_o  = wopen ? (wp + 32'd1 - st0) : 32'd0;
    wire [31:0] occ    = (occ_o > occ_n) ? occ_o : occ_n;

    // ring bank power: the window [wp - pre, wp] is a run of banks
    localparam NBK = 1 << (AW - BW);
    wire             full = on || rd_act || !qempty || (pre >= R - (32'd1 << BW));
    wire [31:0]      wlo  = kp;
    wire [AW-BW-1:0] bhi  = wp[AW-1:BW], blo = wlo[AW-1:BW];
    wire [AW-BW-1:0] nbw  = bhi - blo;                          // banks in the window, minus one
    genvar gb;
    generate for (gb = 0; gb < NBK; gb = gb + 1) begin : g_bank
        wire [AW-BW-1:0] bd = bhi - gb;                           // banks below the write bank (mod NBK)
        assign bank_pe[gb] = full || (bd <= nbw);
    end endgenerate

    wire gen = in_valid || ack || (!rd_act && !qempty) || rwork || ev_wake || ev_tmo || rsync;
    wire gclk;
    clk_gate u_cg (.clk(clk), .en(gen), .gclk(gclk));

`ifndef SYNTHESIS
    reg  [NBK-1:0] pe_q = {NBK{1'b1}};
    reg  [31:0]    pdn = 32'd0;                                 // bank power-downs so far (simulation statistics)
    integer        gi, gj;
`endif
    integer i;
    always @(posedge gclk or negedge rst_n) begin
        if (!rst_n) begin
            wp <= WP0; wfull <= 1'b0; le_v <= 1'b0; last_end <= 32'd0; sstart <= 32'd0; kw <= 32'd0; rp <= 32'd0; hwm <= 32'd0; h <= 8'd0;
            sp <= 8'd0; on <= 1'b0; oq <= 1'b0; rpend <= 1'b0; mark <= 1'b0; ovr <= 1'b0; rd_act <= 1'b0;
            outst <= 1'b0; ack_seen <= 1'b0; qh <= 0; qt <= 0; qc <= 0; qto <= 0; ev_wake <= 1'b0; ev_tmo <= 1'b0;
            q_idx <= 32'd0; q_data <= 16'd0; q_seg <= 1'b0; q_tog <= 1'b0; q_bkl <= 32'd0;
            for (i = 0; i < QD; i = i + 1) begin qs[i] <= 32'd0; qe[i] <= 32'd0; end
        end else begin
            ev_wake <= 1'b0; ev_tmo <= 1'b0;
            // the core is off with nothing in flight: its toggles are back at their reset value, so are ours
            // the core is off: its toggles are at their reset value, so are ours; a sample or end marker that was
            // presented but not taken is presented again after power-up (rp and the queue are unchanged)
            if (!osc_en) begin q_tog <= 1'b0; ack_seen <= 1'b0; outst <= 1'b0; rpend <= 1'b0; mark <= 1'b0; end
            // ---- write side
            if (in_valid) begin
`ifdef SYNTHESIS
                mem[wp[AW-1:0]] <= in_data;
`else
                mem[wp[AW-1:0]] <= bank_pe[wp[AW-1:BW]] ? in_data : 16'($random);
`endif
                wp <= wp + 32'd1;
                h <= h_n;
                wfull <= full_n;
                if (le_v && ((wp - last_end) >= pre)) le_v <= 1'b0;
                if (need && (wp - oldest >= R)) ovr <= 1'b1;          // overwrote a sample still to be delivered
                if (occ > hwm) hwm <= occ;
                if (wopen) begin
                    on <= 1'b1; sstart <= st0; kw <= wp; sp <= 8'd0; ev_wake <= 1'b1;
                    if (!qfull) begin qs[ti] <= st0; qc[ti] <= 1'b0; qto[ti] <= 1'b0; qt <= qt + 1'b1; oq <= 1'b1; end
                    else begin ovr <= 1'b1; oq <= 1'b0; end
                end else if (on) begin
                    sp <= sp_n;
                    if (wclose) begin
                        if (oq) begin qe[tl] <= wp; qc[tl] <= 1'b1; qto[tl] <= c_tmo; end
                        last_end <= wp; le_v <= 1'b1; on <= 1'b0;
                        if (c_tmo) ev_tmo <= 1'b1;
                    end
                end
            end
            // ---- read side
`ifdef SYNTHESIS
            if (re) rq <= mem[rp[AW-1:0]];
`else
            if (re) rq <= bank_pe[rp[AW-1:BW]] ? mem[rp[AW-1:0]] : 16'($random);
            pe_q <= bank_pe;
            for (gi = 0; gi < NBK; gi = gi + 1)
                if (pe_q[gi] && !bank_pe[gi]) begin
                    for (gj = 0; gj < (1 << BW); gj = gj + 1) mem[(gi << BW) + gj] <= 16'($random);
                    pdn <= pdn + 32'd1;
                end
`endif
            if (!rd_act) begin
                if (!qempty) begin rd_act <= 1'b1; rp <= qs[hi]; end
            end else if (!outst) begin
                if (mk) begin
                    q_seg <= 1'b0; q_idx <= hwm; q_data <= {14'd0, qto[hi], ovr}; q_tog <= ~q_tog; outst <= 1'b1;
                    mark <= 1'b1; q_bkl <= 32'd0;
                end else if (re) rpend <= 1'b1;
                else if (rpend && osc_en) begin
                    q_seg <= 1'b1; q_idx <= rp; q_data <= rq; q_tog <= ~q_tog; outst <= 1'b1; rpend <= 1'b0;
                    q_bkl <= wp - rp;
                end
            end
            if (ack) begin
                ack_seen <= ack_s[1];
                outst <= 1'b0;
                if (mark) begin mark <= 1'b0; rd_act <= 1'b0; qh <= qh + 1'b1; end
                else rp <= rp + 32'd1;
            end
        end
    end
endmodule
