// Capture buffer, segmenter and sample delivery, in the front-end clock domain (clk_fe, 3.072 MHz).
//
// Every sample from the PDM front end is written into a ring of 2^AW samples whatever the core is doing, so speech
// that arrives while the core sleeps (pre-roll) or while it transcribes the previous window (backlog) is kept.
// Segments are decided here, live, as samples are written:
//   Rec-gated (strap low): a segment is the stretch while rec is high (start = max(k - pre, end of the previous
//   one) at the first sample k with rec high), cut every cap samples (cap != 0; the next one continues at once).
//   VAD-gated (strap high): golden format 3.4 (docs/golden-format.md 5.1) on the stream's grids, chunks of 1,280
//   samples and VAD frames of 256, both from sample 0 after reset. The VAD flag bundled with sample k (k a multiple
//   of 256) is the decision of frame k / 256 - 1, so the flag of the first sample of chunk c + 1 completes chunk c:
//   on(c) = some frame of chunk c had vad = 1, off(c) = its last frame had vad = 0. There (k = 1280 (c + 1)), while
//   idle and rec is high, on(c) opens a segment at sample max(k - pre - 1280, end of the previous segment): chunk
//   c - 8 with pre = 8 chunks, the pre-roll coming from the ring. An open segment of n = k - start samples closes
//   there when n = cap (250 chunks) or (n >= mn (63 chunks) and off(c)); after a cap close with the VAD still on,
//   the next segment opens at once at k. Rec low (disarm) closes an open segment where it is seen (stream end).
// Segments queue (QD deep) and are delivered to the core in order, one sample per handshake: {index, sample,
// seg = 1} in registers that hold still until the core answers (bundled data), announced by a toggle (q_tog) and
// taken with a toggle back (ack_t). After a segment's last sample comes one end marker: seg = 0, index = the
// high-water mark of the buffer (most unread samples so far), data[0] = overrun (unread samples were overwritten,
// or a segment found the queue full; sticky).
// With every sample also goes q_bkl, the backlog when it is presented: samples written but not yet delivered (the
// firmware's load guard reads it; rtl/core/fw/stream.c).
// Oscillator: the core asks for deep sleep with a toggle (dq_t). Unless the core has something to do (a segment
// queued or being delivered, or rec differs from rexp) the core oscillator is stopped (osc_en low); when there is,
// it is started again and the core woken with a toggle (wk_t): exactly one wake per request, so the core is
// asleep whenever the oscillator is stopped.
// pre, cap, mn and rexp are quasi-static: the core writes pre, cap and mn while the front end is off and rexp before it
// toggles dq_t. The ring is a single-port memory: a write per sample, reads in the other cycles.
// Ring banks (power fix 3, reports/p1-power.md section 7): 2^(AW-BW) banks of 2^BW samples, the bank being the high
// address bits, so consecutive samples share a bank and the pre-roll window is a run of banks. While no segment is
// open, queued or being delivered, only the banks holding the samples a new segment could start from (the last
// pre + 1,280, the onset chunk included) and the one being written are powered (bank_pe); every bank is powered again
// when a segment opens. A bank keeps its contents only while powered: in simulation a bank is filled with garbage
// when it is powered down and a read of an unpowered bank returns garbage, so a stale read fails the bit-exact check.
module imh_capture #(
    parameter AW = 18,           // ring of 2^AW samples (2^AW x 16 bit)
    parameter BW = 11,           // banks of 2^BW samples (4 KB)
    parameter QD = 4             // queued segments (power of two)
) (
    input  wire               clk,
    input  wire               rst_n,       // front-end reset (asynchronous assert, synchronous release)
    input  wire               in_valid,
    input  wire signed [15:0] in_data,
    input  wire               in_vad,      // VAD output before this sample was analysed
    input  wire               rec_i,       // rec pin (synchronised here)
    input  wire               vg_i,        // VAD-gate strap (synchronised here)
    input  wire [31:0]        pre,
    input  wire [31:0]        cap,
    input  wire [31:0]        mn,          // VAD-gated: no pause close before a segment holds this many samples
    input  wire               rexp,
    input  wire               ack_t,       // from the core clock domain (synchronised here)
    input  wire               dq_t,
    output reg  [31:0]        q_idx,
    output reg  [15:0]        q_data,
    output reg                q_seg,
    output reg  [31:0]        q_bkl,       // with a sample: backlog (written, not yet delivered) when presented
    output reg                q_tog,
    output reg                wk_t,
    output reg                osc_en,
    output wire [(1 << (AW - BW)) - 1:0] bank_pe   // ring bank power enables (1 = powered)
);
    localparam QB = $clog2(QD);
    localparam [31:0] R = 32'd1 << AW;

    // ---- free-running: synchronisers and the oscillator control
    reg  [1:0] rec_s, vg_s, ack_s, dq_s;
    reg        dq_seen, slp;
    wire       rec = rec_s[1], vg = vg_s[1];
    wire       dq  = (dq_s[1] != dq_seen);
    reg        rd_act, outst, ack_seen;
    reg  [QB:0] qh, qt;
    wire       qempty = (qh == qt);
    wire       ev = rd_act || !qempty || outst || (rec != rexp);
    always @(posedge clk or negedge rst_n)
        if (!rst_n) begin
            rec_s <= 2'd0; vg_s <= 2'd0; ack_s <= 2'd0; dq_s <= 2'd0; dq_seen <= 1'b0; slp <= 1'b0;
            wk_t <= 1'b0; osc_en <= 1'b1;
        end else begin
            rec_s <= {rec_s[0], rec_i}; vg_s <= {vg_s[0], vg_i}; ack_s <= {ack_s[0], ack_t}; dq_s <= {dq_s[0], dq_t};
            if (dq) begin
                dq_seen <= dq_s[1];
                if (ev) wk_t <= ~wk_t;                       // something to do already: wake at once
                else begin osc_en <= 1'b0; slp <= 1'b1; end  // stop the core oscillator
            end else if (slp && ev) begin
                osc_en <= 1'b1; slp <= 1'b0; wk_t <= ~wk_t;
            end
        end

    // ---- everything else runs on a gated clock: once per sample while listening, every clock while delivering
    wire ack = (ack_s[1] != ack_seen);

    reg  [15:0] mem [0:(1 << AW) - 1];
    reg  [15:0] rq;
    reg  [10:0] cpos;          // wp mod 1280: position in the chunk grid
    reg         onc;           // a frame of the current chunk had vad = 1 (flags seen so far)
    reg  [31:0] wp, last_end, sstart, rp, hwm;
    reg         on, oq, rpend, mark, ovr;
    reg  [31:0] qs [0:QD-1];
    reg  [31:0] qe [0:QD-1];
    reg  [QD-1:0] qc;
    localparam [QB:0] QDV = QD;
    wire [QB-1:0] hi = qh[QB-1:0], ti = qt[QB-1:0], tl = ti - 1'b1;
    wire        qfull = ((qt - qh) == QDV);

    // segmenter, evaluated for the sample being written (index wp)
    // rec-gated
    wire [31:0] kp     = (wp > pre) ? (wp - pre) : 32'd0;
    wire [31:0] st0    = (kp > last_end) ? kp : last_end;
    wire        capped = on && (cap != 32'd0) && ((wp - sstart) == cap);
    // VAD-gated (format 3.4): decided at the first sample of every chunk, for the chunk before it
    wire        cb     = (cpos == 11'd0) && (wp != 32'd0);
    wire        on_c   = onc || in_vad;                           // the flag of sample wp is the chunk's last frame
    wire        off_c  = !in_vad;
    wire [31:0] pr1    = pre + 32'd1280;                          // pre-roll chunks and the onset chunk
    wire [31:0] kv     = (wp > pr1) ? (wp - pr1) : 32'd0;
    wire [31:0] stv    = (kv > last_end) ? kv : last_end;
    wire [31:0] slen   = wp - sstart;
    wire        vcap   = cb && (cap != 32'd0) && (slen == cap);
    wire        vopen  = vg && !on && rec && cb && on_c;
    wire        vclose = vg && on && (!rec || vcap || (cb && slen >= mn && off_c));
    wire        vnext  = vcap && !off_c && rec;                   // cap close with the VAD on: the next opens at once
    // reader
    wire        mk     = rd_act && !outst && !rpend && qc[hi] && (rp == qe[hi]);       // end marker due
    wire        re     = rd_act && !outst && !rpend && !mk && (rp != wp) && !in_valid;  // read the next sample
    wire        rwork  = rd_act && !outst && (rpend || (rp != wp) || (qc[hi] && (rp == qe[hi])));
    // buffer use: unread samples of queued segments, counting the one being written
    wire        need   = rd_act || !qempty;
    wire [31:0] oldest = rd_act ? rp : qs[hi];
    wire [31:0] occ_n  = need ? (wp + 32'd1 - oldest) : 32'd0;
    wire [31:0] occ_o  = vg ? (vopen ? (wp + 32'd1 - stv) : 32'd0)   // a segment opens with its pre-roll
                            : ((!on && rec) ? (wp + 32'd1 - st0) : 32'd0);
    wire [31:0] occ    = (occ_o > occ_n) ? occ_o : occ_n;

    // ring bank power: the window [wp - pre - 1280, wp] is a run of banks, from the bank of its oldest sample to
    // the bank of wp (mod the bank count; the window is shorter than the ring unless pre is set absurdly high)
    localparam NBK = 1 << (AW - BW);
    wire [31:0]      win  = pre + 32'd1280;
    wire             full = on || rd_act || !qempty || (win >= R - (32'd1 << BW));
    wire [31:0]      wlo  = (wp > win) ? (wp - win) : 32'd0;
    wire [AW-BW-1:0] bhi  = wp[AW-1:BW], blo = wlo[AW-1:BW];
    wire [AW-BW-1:0] nbw  = bhi - blo;                          // banks in the window, minus one
    genvar gb;
    generate for (gb = 0; gb < NBK; gb = gb + 1) begin : g_bank
        wire [AW-BW-1:0] bd = bhi - gb;                           // banks below the write bank (mod NBK)
        assign bank_pe[gb] = full || (bd <= nbw);
    end endgenerate

    wire gen = in_valid || ack || (!rd_act && !qempty) || rwork;
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
            wp <= 32'd0; last_end <= 32'd0; sstart <= 32'd0; rp <= 32'd0; hwm <= 32'd0;
            on <= 1'b0; oq <= 1'b0; rpend <= 1'b0; mark <= 1'b0; ovr <= 1'b0; rd_act <= 1'b0; outst <= 1'b0;
            ack_seen <= 1'b0; qh <= 0; qt <= 0; qc <= 0;
            q_idx <= 32'd0; q_data <= 16'd0; q_seg <= 1'b0; q_tog <= 1'b0; q_bkl <= 32'd0; cpos <= 11'd0; onc <= 1'b0;
            for (i = 0; i < QD; i = i + 1) begin qs[i] <= 32'd0; qe[i] <= 32'd0; end
        end else begin
            // ---- write side
            if (in_valid) begin
`ifdef SYNTHESIS
                mem[wp[AW-1:0]] <= in_data;
`else
                mem[wp[AW-1:0]] <= bank_pe[wp[AW-1:BW]] ? in_data : 16'($random);
`endif
                wp <= wp + 32'd1;
                cpos <= (cpos == 11'd1279) ? 11'd0 : cpos + 11'd1;
                if (cb) onc <= 1'b0;                                       // chunk c is decided now
                else if (wp[7:0] == 8'd0 && wp != 32'd0) onc <= onc | in_vad;    // frame wp / 256 - 1
                if (need && (wp - oldest >= R)) ovr <= 1'b1;          // overwrote a sample still to be delivered
                if (occ > hwm) hwm <= occ;
                if (!vg) begin                                        // rec-gated
                    if (!on) begin
                        if (rec) begin
                            on <= 1'b1; sstart <= st0;
                            if (!qfull) begin qs[ti] <= st0; qc[ti] <= 1'b0; qt <= qt + 1'b1; oq <= 1'b1; end
                            else begin ovr <= 1'b1; oq <= 1'b0; end
                        end
                    end else if (capped || !rec) begin
                        if (oq) begin qe[tl] <= wp; qc[tl] <= 1'b1; end
                        last_end <= wp;
                        if (capped && rec) begin
                            sstart <= wp;
                            if (!qfull) begin qs[ti] <= wp; qc[ti] <= 1'b0; qt <= qt + 1'b1; oq <= 1'b1; end
                            else begin ovr <= 1'b1; oq <= 1'b0; end
                        end else on <= 1'b0;
                    end
                end else if (vopen) begin                             // VAD-gated, format 3.4
                    on <= 1'b1; sstart <= stv;
                    if (!qfull) begin qs[ti] <= stv; qc[ti] <= 1'b0; qt <= qt + 1'b1; oq <= 1'b1; end
                    else begin ovr <= 1'b1; oq <= 1'b0; end
                end else if (vclose) begin
                    if (oq) begin qe[tl] <= wp; qc[tl] <= 1'b1; end
                    last_end <= wp;
                    if (vnext) begin
                        sstart <= wp;
                        if (!qfull) begin qs[ti] <= wp; qc[ti] <= 1'b0; qt <= qt + 1'b1; oq <= 1'b1; end
                        else begin ovr <= 1'b1; oq <= 1'b0; end
                    end else on <= 1'b0;
                end
            end
            // ---- read side
`ifdef SYNTHESIS
            if (re) rq <= mem[rp[AW-1:0]];
`else
            if (re) rq <= bank_pe[rp[AW-1:BW]] ? mem[rp[AW-1:0]] : 16'($random);
            // a bank powered down at the last edge loses its contents
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
                    q_seg <= 1'b0; q_idx <= hwm; q_data <= {15'd0, ovr}; q_tog <= ~q_tog; outst <= 1'b1; mark <= 1'b1;
                    q_bkl <= 32'd0;
                end else if (re) rpend <= 1'b1;
                else if (rpend) begin
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
