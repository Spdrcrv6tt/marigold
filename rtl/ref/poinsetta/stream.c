// Moonshine Streaming Tiny on the Poinsetta core: the program the controller runs (docs/golden-format.md 3.6,
// reference IntStream / EncoderState / segments in sw/golden/stream_np.py, bit for bit).
// Audio arrives sample by sample from the capture buffer; every 1,280 samples (a chunk) the frontend makes 16 rows of
// 80 samples -> 8 conv1 rows -> 4 encoder input frames, and the encoder releases every frame whose look-ahead
// exists. At the update slots of the schedule (golden format 3.6, docs/golden-format.md 5.2: SCHED_U, SCHED_R) while
// the segment continues, and at the segment end, the decoder updates the segment's text against the memory frames
// released so far: a refresh (and the segment end) decodes the whole segment again from BOS, a tail update keeps the
// last refresh's first ids with their self-attention K/V and decodes only the rest. The update's text goes out of
// the UART (partial updates end in '\r', the segment's final text in '\n'). Segments come from the capture buffer
// (format 3.4 when VAD-gated); partial updates are skipped while the capture backlog is high (LG_BKL): see model_main.
// Weight products: the fan-out array (FAN_GO). Attention: the batch MAC array with RAM operands (MAC_GO; cross
// attention reads int8 K and V, the encoder's band attention reads K/V rings). Per element: the per-element unit.
// Storage (addresses in 16-bit elements unless they say bytes): rows are frame-major (a frame's 320 values
// contiguous); the 25-bit residual is two planes (low 16 bits, high 16 bits) PL apart; K for attention is
// feature-major [320][keys] (keys in the array's lanes), V head-major [8][keys][48].
#include "hal.h"
#include "layout.h"
#define B LANES
#define NLY 6
#define D 320
#define NH 8
#define HD 40
#define HP 48
#define FFE 1280
#define FFD 1280
#define RING 32                   // encoder K/V/q/x rings (frames); > 16 + 4 + batch
#define FRAME 80
#define CHUNK 1280
// Update schedule (golden format 3.6, docs/golden-format.md 5.2): pairs (until, interval) by the segment's chunk
// count; the slot after slot k (0 at the segment start) is k + the interval of the first pair with k < until.
// SCHED_U: update slots, SCHED_R: refresh slots (decoded from BOS; their K/V kept); TAIL_KEEP: a tail update keeps the
// last refresh's ids but its last TAIL_KEEP. -DUPD=n: format 3.5, a refresh every n chunks and nothing else.
#ifdef UPD
#define SCHED_U 65535, UPD
#define SCHED_R 65535, UPD
#define TAIL_KEEP 0
#endif
#ifndef SCHED_U
#define SCHED_U 48, 2, 96, 3, 160, 4, 65535, 8
#endif
#ifndef SCHED_R
#define SCHED_R 96, 16, 192, 24, 65535, 32
#endif
#ifndef TAIL_KEEP
#define TAIL_KEEP 4
#endif
#define I16M 32767
#define I8M 127
#define RES_SAT ((1 << 24) - 1)
#define BOS 1
#define EOS 2
#define MAX_TOK 448
#ifndef STREAM_MAX_SEC
#define STREAM_MAX_SEC 20
#endif
#ifndef TRACE_MEM
#define TRACE_MEM 0               // 1: one hash per memory row (encoder output frame) even with the trace strap off
#endif
#ifndef TRACE_STEPS
#define TRACE_STEPS 3             // decoder steps per update whose internals are traced
#endif
#define UP(n) (((n) + B - 1) & ~(u32)(B - 1))
#define MAXF UP(50 * STREAM_MAX_SEC + 16)   // memory frames per segment
#define MTC ((13 * 50 * STREAM_MAX_SEC + 99) / 100)   // token cap at the segment cap (130 at 20 s): the decoder cache's rows
#define P16(e) ((i16 *)(ram + 2u * (e)))
#define S_FE_LIN 0
#define S_FE_C1 1
#define S_FE_C2 2
#define S_ENC(i, j) (3 + 6 * (i) + (j))          // j: 0 q 1 k 2 v 3 o 4 fc1 5 fc2
#define S_DEC(i, j) (39 + 10 * (i) + (j))        // j: 0 q 1 k 2 v 3 o 4 cq 5 ck 6 cv 7 co 8 fc1 9 fc2
#define S_HEAD 99
#define N_CMVN 0
#define N_ENC(i, j) (1 + 2 * (i) + (j))          // j: 0 ln1 1 ln2
#define N_ENCLN 13
#define N_DEC(i, j) (14 + 3 * (i) + (j))         // j: 0 ln1 1 ln2 2 ln3
#define N_DECLN 32
#define T_GELU 1
#define T_SILU 2
#define T_ASINH 3
static const int FUT[NLY] = {4, 4, 0, 0, 4, 4};
static const signed char FQ_E[NLY] = {F_ENC0_Q, F_ENC1_Q, F_ENC2_Q, F_ENC3_Q, F_ENC4_Q, F_ENC5_Q};
static const signed char FK_E[NLY] = {F_ENC0_K, F_ENC1_K, F_ENC2_K, F_ENC3_K, F_ENC4_K, F_ENC5_K};
static const signed char FF_E[NLY] = {F_ENC0_FC1, F_ENC1_FC1, F_ENC2_FC1, F_ENC3_FC1, F_ENC4_FC1, F_ENC5_FC1};
static const signed char FQ_D[NLY] = {F_DEC0_Q, F_DEC1_Q, F_DEC2_Q, F_DEC3_Q, F_DEC4_Q, F_DEC5_Q};
static const signed char FK_D[NLY] = {F_DEC0_K, F_DEC1_K, F_DEC2_K, F_DEC3_K, F_DEC4_K, F_DEC5_K};
static const signed char FCQ[NLY] = {F_DEC0_CQ, F_DEC1_CQ, F_DEC2_CQ, F_DEC3_CQ, F_DEC4_CQ, F_DEC5_CQ};
static const signed char FCK[NLY] = {F_DEC0_CK, F_DEC1_CK, F_DEC2_CK, F_DEC3_CK, F_DEC4_CK, F_DEC5_CK};
static const signed char FCK8[NLY] = {F_DEC0_CK8, F_DEC1_CK8, F_DEC2_CK8, F_DEC3_CK8, F_DEC4_CK8, F_DEC5_CK8};
static const signed char FCV[NLY] = {F_DEC0_CV, F_DEC1_CV, F_DEC2_CV, F_DEC3_CV, F_DEC4_CV, F_DEC5_CV};
static const signed char FCV8[NLY] = {F_DEC0_CV8, F_DEC1_CV8, F_DEC2_CV8, F_DEC3_CV8, F_DEC4_CV8, F_DEC5_CV8};
static const signed char FF_D[NLY] = {F_DEC0_FC1, F_DEC1_FC1, F_DEC2_FC1, F_DEC3_FC1, F_DEC4_FC1, F_DEC5_FC1};
static const signed char FPR_D[NLY] = {F_DEC0_PROD, F_DEC1_PROD, F_DEC2_PROD, F_DEC3_PROD, F_DEC4_PROD, F_DEC5_PROD};

// trace tags: kind << 20 | index (rtl/top/tb/check_ts.py has the same table)
enum { K_CMVN = 1, K_ASINH, K_LIN, K_C1, K_FEAT, K_ENC = 10, K_MEM = 20, K_CK8 = 30, K_CV8 = 40, K_DEC = 100,
       K_UPD = 200, K_TOK, K_UEND, K_SEG, K_DONE, K_CAP, K_SKIP, K_SEGE, K_SLOT };
enum { DN_EMB, DN_LN1, DN_QROT, DN_KROT, DN_V, DN_ATTN, DN_RES1, DN_LN2, DN_CQ, DN_CATTN, DN_RES2, DN_LN3, DN_FC1,
       DN_SILU, DN_PROD, DN_RES3, DN_LN, DN_LOGITS };
#define TT(k, i) (((u32)(k) << 20) | (u32)(i))
#define DI(u, t, l) (((u32)(u) << 12) | ((u32)(t) << 4) | (u32)(l))
#define HMUL 16777619u
static int tron;

// ------------------------------------------------------------------------------------------ small helpers
static inline i64 rr64(i64 x, int s) { return s <= 0 ? x << (-s) : (x + ((i64)1 << (s - 1))) >> s; }
static inline i32 sat(i64 v, i32 m) { return v > m ? m : v < -m ? -m : (i32)v; }
// value f of a row-scaled table row (emb, pos) being read from the weight stream: packed two per byte, low nibble first,
// when the image says so (EMB_Q4 / POS_Q4; the byte is read at even f and kept in *b), else one int8 per byte
static inline i64 rowq(int q4, u32 f, u32 *b) {
    if (!q4) return (i64)(i8)fl_u8();
    if (!(f & 1)) { *b = fl_u8(); return (i64)((i32)(i8)(*b << 4) >> 4); }
    return (i64)((i32)(i8)*b >> 4);
}
static u32 brk_e, top_e, end_e, hiw_e;
static void use(void) { u32 u = brk_e + (end_e - top_e); if (u > hiw_e) hiw_e = u; if (brk_e > top_e) hal_fail(1); }
static u32 al(u32 n) { u32 e = brk_e; brk_e += UP(n); use(); return e; }
static u32 alt(u32 n) { top_e -= UP(n); use(); return top_e; }
static u32 fl_u16(void) { u32 a = fl_u8(); return a | (fl_u8() << 8); }
static u32 fl_u32(void) { u32 a = fl_u16(); return a | (fl_u16() << 16); }
static inline i32 rd25(u32 lo, u32 pl) { return (i32)(((u32)(u16)P16(lo)[pl] << 16) | (u16)P16(lo)[0]); }
static inline void wr25(u32 lo, u32 pl, i32 v) { P16(lo)[0] = (i16)v; P16(lo)[pl] = (i16)(v >> 16); }

static void tr16(u32 tag, u32 e, u32 n) {
    if (!tron) return;
    u32 h = 0; for (u32 i = 0; i < n; i++) h = h * HMUL + (u32)(i32)P16(e)[i];
    trace(tag, h);
}
static void tr25(u32 tag, u32 lo, u32 pl, u32 n) {
    if (!tron) return;
    u32 h = 0; for (u32 i = 0; i < n; i++) h = h * HMUL + (u32)rd25(lo + i, pl);
    trace(tag, h);
}

// ------------------------------------------------------------------------------------------ per-element unit
static void v_copy(u32 src, u32 sstr, u32 srow, u32 dst, u32 dstr, u32 drow, u32 n, u32 rows) {
    VEC->src = src; VEC->sstr = sstr; VEC->srow = srow; VEC->dst = dst; VEC->dstr = dstr; VEC->drow = drow;
    VEC->n = n; VEC->rows = rows;
    VEC_GO(VEC_COPY);
}
// to int8: dst byte = sat(rshift_round(src, sh), 127); dst, dstr, drow in bytes
static void v_copy8(u32 src, u32 sstr, u32 srow, u32 dstb, u32 dstr, u32 drow, u32 n, u32 rows, int sh) {
    VEC->src = src; VEC->sstr = sstr; VEC->srow = srow; VEC->dst = dstb; VEC->dstr = dstr; VEC->drow = drow;
    VEC->n = n; VEC->rows = rows; VEC->p0 = (u32)sh;
    VEC_GO(VEC_COPY | VEC_B8);
}
static void v_lut(u32 a, u32 n, u32 rows, u32 row, int f, int fo, int tab, int tails, int r4) {
    VEC->src = a; VEC->sstr = 1; VEC->srow = row; VEC->dst = a; VEC->dstr = 1; VEC->drow = row; VEC->n = n; VEC->rows = rows;
    VEC->p0 = (u32)f | ((u32)fo << 4) | ((u32)tab << 8) | ((u32)tails << 10) | ((u32)r4 << 13);
    VEC_GO(VEC_LUT);
}
static void v_norm(u32 src, u32 pl, int wide, u32 dst, u32 n, int idx) {
    VEC->p0 = n; VEC->p1 = (u32)NORMS[idx].eps;
    VEC_GO(VEC_NSTAT);
    VEC->src = src; VEC->sstr = 1; VEC->srow = 0; VEC->dst = dst; VEC->dstr = 1; VEC->drow = 0;
    VEC->n = n; VEC->rows = 1; VEC->p1 = pl;
    VEC->p3 = (u32)(12 + NORMS[idx].fg - NORMS[idx].fout) << 8; VEC->faddr = NORMS[idx].off;
    VEC_GO(VEC_NORM | VEC_AUTO | (wide ? VEC_WIDE : 0));
}
// LayerNorm of a 25-bit row (planes lo, lo + pl)
static void norm25(u32 lo, u32 pl, u32 dst, int idx) {
    VEC->src = lo; VEC->sstr = 1; VEC->n = D; VEC->rows = 1; VEC->p1 = pl;
    VEC_GO(VEC_STAT | VEC_CLR | VEC_WIDE);
    v_norm(lo, pl, 1, dst, D, idx);
}
// norm of a 16-bit row of n values
static void norm16(u32 src, u32 n, u32 dst, int idx) {
    VEC->src = src; VEC->sstr = 1; VEC->n = n; VEC->rows = 1;
    VEC_GO(VEC_STAT | VEC_CLR);
    v_norm(src, 0, 0, dst, n, idx);
}
static void v_rope(u32 v, u32 pos) {
    VEC->src = v; VEC->sstr = 2; VEC->srow = HD; VEC->p1 = 1; VEC->n = 16; VEC->rows = NH;
    VEC->faddr = OFF_ROPE + 64 * pos;
    VEC_GO(VEC_ROPE);
}
static void v_mul(u32 a, u32 boff, u32 dst, u32 n, int sh) {
    VEC->src = a; VEC->sstr = 1; VEC->srow = 0; VEC->dst = dst; VEC->dstr = 1; VEC->drow = 0; VEC->n = n; VEC->rows = 1;
    VEC->p0 = (u32)sh; VEC->p1 = boff;
    VEC_GO(VEC_MUL);
}

// rows of n products: dst row r = a row r (at a + r * srow) times b (at the same element + boff)
static void v_mulr(u32 a, u32 srow, u32 boff, u32 dst, u32 drow, u32 n, u32 rows, int sh) {
    VEC->src = a; VEC->sstr = 1; VEC->srow = srow; VEC->dst = dst; VEC->dstr = 1; VEC->drow = drow; VEC->n = n; VEC->rows = rows;
    VEC->p0 = (u32)sh; VEC->p1 = boff;
    VEC_GO(VEC_MUL);
}
// ------------------------------------------------------------------------------------------ fan-out array
static void fan(int s, u32 x, u32 xs, u32 nc, u32 y, u32 ys, u32 yp, u32 mode) {
    const site_t *S = &SITES[s];
    FAN->faddr = S->off; FAN->x = x; FAN->xs = xs; FAN->nin = S->nin; FAN->nout = S->nout; FAN->nc = nc;
    FAN->y = y; FAN->ys = ys; FAN->yp = yp;
    FAN_GO((S->wbits == 4 ? FAN_W4 : S->wbits == 16 ? FAN_W16 : 0) | mode);
}
static void fan_rows(int s, u32 x, u32 xs, u32 n, u32 y, u32 ys, u32 yp, u32 mode) {
    for (u32 c = 0; c < n; c += FAN_CMAX) fan(s, x + c * xs, xs, n - c < FAN_CMAX ? n - c : FAN_CMAX, y + c * ys, ys, yp, mode);
}

// ------------------------------------------------------------------------------------------ attention
// scratch: raw scores (3 planes of [8][SCP]), e (int32 [8][nk]), reciprocals (u32 [8]), mixed output [8][HP]
static u32 SC, SCP, EB, RCB, MIX;
static void attn_scratch(u32 nk) {
    SCP = UP(nk > 64 ? nk : 64); SC = al(3 * NH * SCP); EB = al(2 * NH * (nk > 64 ? nk : 64)); RCB = al(2 * NH); MIX = al(NH * HP);
}
static void scores(u32 wq, u32 K, u32 kstr, u32 kb, u32 y, u32 x8) {
    MAC->w = wq; MAC->wi = 2; MAC->wo = 2 * HD; MAC->xi = kstr; MAC->xo = HD * kstr; MAC->xw = 0;
    MAC->nin = HD; MAC->nout = NH; MAC->ys = SCP; MAC->yp = NH * SCP; MAC->x = K + kb; MAC->y = y;
    MAC_GO(MAC_W_RAM16 | MAC_O_RAW | x8);
}
// softmax over nk scores starting at column s0, then mixing with V (VH layout, head stride vhs; ring if xw)
static void soft_mix(u32 s0, u32 nk, int sh, u32 VH, u32 vhs, u32 xw, u32 xs0, int up, u32 x8, u32 out) {
    VEC->src = SC + s0; VEC->sstr = 1; VEC->srow = SCP; VEC->p1 = NH * SCP; VEC->n = nk; VEC->rows = NH;
    VEC->dst = EB; VEC->dstr = 2; VEC->drow = 2 * nk; VEC->p0 = (u32)SM_C | ((u32)sh << 16); VEC->p2 = RCB;
    VEC_GO(VEC_SOFTMAX);
    MAC->w = 2u * EB; MAC->wi = 4; MAC->wo = 4 * nk; MAC->xi = HP; MAC->xo = vhs; MAC->xw = xw; MAC->xs0 = xs0;
    MAC->nin = nk; MAC->nout = NH; MAC->ys = HP; MAC->yp = 0; MAC->h = 2u * RCB; MAC->hsh = 44 - up;
    for (u32 g = 0; g < HP; g += B) { MAC->x = VH + g; MAC->y = MIX + g; MAC_GO(MAC_W_RAM32 | MAC_O_S16 | MAC_HDRRAM | x8); }
    MAC->xw = 0; MAC->xs0 = 0;
    v_copy(MIX, 1, HP, out, 1, HD, HD, NH);
}

// ------------------------------------------------------------------------------------------ segment state
static u32 CHB, Y80, HB, CB, XR, XPL, QR[NLY], KR[NLY], VR[NLY], TA, TB, TC, MEMB;
static u32 CK8b[NLY], CV8b[NLY];          // byte addresses: K int8 [320][MAXF], V int8 [8][MAXF][HP]
static u32 KC[NLY], VC[NLY], dMT, dMTp;    // decoder self-attention cache, kept for the segment (tail updates reuse
                                           // a refresh's rows): K [320][UP(MTC)], V [8][MTC][HP]
static u32 n_lin, n_c1, n_c2, nT[NLY + 1], kvn[NLY], done[NLY], n_mem;

static void seg_init(void) {
    CHB = al(CHUNK); Y80 = al(16 * FRAME);
    HB = al(20 * D); CB = al(12 * 2 * D);
    XPL = RING * D; XR = al(2 * XPL);
    for (int i = 0; i < NLY; i++) { QR[i] = al(RING * D); KR[i] = al(D * RING); VR[i] = al(NH * RING * HP); }
    TA = al(FAN_CMAX * D); TB = al(FAN_CMAX * D); TC = al(FAN_CMAX * FFE); MEMB = al(FAN_CMAX * D);
    attn_scratch(64);                       // the encoder's band attention (the decoder makes its own)
    for (int i = 0; i < NLY; i++) { CK8b[i] = 2u * alt(D * MAXF / 2); CV8b[i] = 2u * alt(NH * MAXF * HP / 2); }
    dMT = MTC; dMTp = UP(MTC);
    for (int i = 0; i < NLY; i++) { KC[i] = al(D * dMTp); VC[i] = al(NH * MTC * HP); }
    for (u32 e = 0; e < 4 * D; e++) P16(HB)[e] = 0;
    for (u32 e = 0; e < 4 * 2 * D; e++) P16(CB)[e] = 0;
    n_lin = n_c1 = n_c2 = n_mem = 0;
    for (int i = 0; i <= NLY; i++) nT[i] = 0;
    for (int i = 0; i < NLY; i++) { kvn[i] = 0; done[i] = 0; }
}
#define SLOT(t) ((t) % RING)

// encoder output frames [n_mem, n_mem + n): adapter (norm, learned position), int8 cross K/V
static void memory_rows(u32 t0, u32 n) {
    for (u32 c = 0; c < n; c++) {
        u32 t = t0 + c, row = MEMB + c * D;
        norm25(XR + SLOT(t) * D, XPL, TA, N_ENCLN);
        fl_seek(OFF_POS_REC + 4 * t);
        i64 pm = (i64)fl_u16(); int psh = (int)fl_u8();
        fl_seek(OFF_POS_Q + (POS_Q4 ? D / 2 : D) * t);
        u32 pb = 0;
        for (u32 f = 0; f < D; f++) {
            i64 pe = rr64(rowq(POS_Q4, f, &pb) * pm, psh);
            P16(row)[f] = (i16)sat(rr64(P16(TA)[f], F_ENC_LN - F_MEM) + pe, I16M);
        }
        if (tron || TRACE_MEM) {            // TRACE_MEM: memory rows also with the trace strap off (regression runs)
            u32 h = 0; for (u32 f = 0; f < D; f++) h = h * HMUL + (u32)(i32)P16(row)[f];
            trace(TT(K_MEM, t), h);
        }
    }
    for (int i = 0; i < NLY; i++) {
        fan(S_DEC(i, 5), MEMB, D, n, TA, D, 0, FAN_O_S16);
        fan(S_DEC(i, 6), MEMB, D, n, TB, D, 0, FAN_O_S16);
        for (u32 c = 0; c < n; c++) {
            u32 t = t0 + c;
            v_copy8(TA + c * D, 1, 0, CK8b[i] + t, MAXF, 0, D, 1, FCK[i] - FCK8[i]);
            v_copy8(TB + c * D, 1, HD, CV8b[i] + t * HP, 1, MAXF * HP, HD, NH, FCV[i] - FCV8[i]);
            if (tron) {
                u32 hk = 0, hv = 0;
                for (u32 f = 0; f < D; f++) hk = hk * HMUL + (u32)(i32)(i8)ram[CK8b[i] + f * MAXF + t];
                for (u32 h = 0; h < NH; h++) for (u32 d = 0; d < HD; d++)
                    hv = hv * HMUL + (u32)(i32)(i8)ram[CV8b[i] + h * MAXF * HP + t * HP + d];
                trace(TT(K_CK8 + i, t), hk); trace(TT(K_CV8 + i, t), hv);
            }
        }
    }
}

// band attention of encoder layer i for query frame t over keys [j0, j0 + nk)
static void band_attn(int i, u32 t, u32 j0, u32 nk, u32 out) {
    for (u32 m = 0; m < 2; m++)                     // all 32 ring slots, twice: a window that wraps reads straight
        for (u32 kb = 0; kb < RING; kb += B) scores(2u * (QR[i] + SLOT(t) * D), KR[i], RING, kb, SC + m * RING + kb, 0);
    soft_mix(SLOT(j0), nk, FQ_E[i] + FK_E[i] + 4, VR[i], RING * HP, RING * HP, SLOT(j0) * HP, 0, 0, out);
}

// layer i: K, V, q of new input frames, then every frame whose look-ahead exists. K/V are computed at most 16
// frames ahead of the next frame to release, so the 32-frame rings always hold its whole window.
static void enc_layer(int i, int final) {
    u32 T = nT[i], fut = (u32)FUT[i];
    u32 ready = final ? T : (T > fut ? T - fut : 0);
    for (;;) {
        u32 lim = done[i] + 16 < T ? done[i] + 16 : T;
        while (kvn[i] < lim) {                      // new inputs, in runs that do not wrap the rings
            u32 t = kvn[i], n = lim - t, s = SLOT(t);
            if (n > FAN_CMAX) n = FAN_CMAX;
            if (n > RING - s) n = RING - s;
            for (u32 c = 0; c < n; c++) norm25(XR + (s + c) * D, XPL, TA + c * D, N_ENC(i, 0));
            fan(S_ENC(i, 0), TA, D, n, QR[i] + s * D, D, 0, FAN_O_S16);
            fan(S_ENC(i, 1), TA, D, n, TB, D, 0, FAN_O_S16);
            fan(S_ENC(i, 2), TA, D, n, TC, D, 0, FAN_O_S16);
            for (u32 c = 0; c < n; c++) {
                v_copy(TB + c * D, 1, 0, KR[i] + s + c, RING, 0, D, 1);
                v_copy(TC + c * D, 1, HD, VR[i] + (s + c) * HP, 1, RING * HP, HD, NH);
            }
            kvn[i] += n;
        }
        if (done[i] >= ready) break;
        u32 t0 = done[i], n = ready - t0, s = SLOT(t0);
        if (n > FAN_CMAX) n = FAN_CMAX;
        if (n > RING - s) n = RING - s;
        while (n > 1 && (t0 + n - 1 + fut < T ? t0 + n - 1 + fut : T - 1) >= kvn[i]) n--;   // keys must exist
        for (u32 c = 0; c < n; c++) {
            u32 t = t0 + c, j0 = t > 16 ? t - 16 : 0, j1 = t + FUT[i] < T ? t + FUT[i] : T - 1;
            band_attn(i, t, j0, j1 - j0 + 1, TA + c * D);
        }
        u32 xr = XR + s * D;
        fan(S_ENC(i, 3), TA, D, n, xr, D, XPL, FAN_O_RES | FAN_RESACC);
        for (u32 c = 0; c < n; c++) norm25(xr + c * D, XPL, TB + c * D, N_ENC(i, 1));
        fan(S_ENC(i, 4), TB, D, n, TC, FFE, 0, FAN_O_S16);
        v_lut(TC, FFE, n, FFE, FF_E[i], FF_E[i], T_GELU, 1, 0);
        fan(S_ENC(i, 5), TC, FFE, n, xr, D, XPL, FAN_O_RES | FAN_RESACC);
        for (u32 c = 0; c < n; c++) tr25(TT(K_ENC + i, t0 + c), xr + c * D, XPL, D);
        done[i] += n;
    }
    nT[i + 1] = done[i];
}

// one chunk of nf frames of 80 samples in CHB (final: the segment's last, zero-padded)
static void chunk(u32 nf, int final) {
    for (u32 r = 0; r < nf; r++) {
        norm16(CHB + r * FRAME, FRAME, Y80 + r * FRAME, N_CMVN);
        tr16(TT(K_CMVN, n_lin + r), Y80 + r * FRAME, FRAME);
    }
    if (nf) v_lut(Y80, FRAME, nf, FRAME, F_FE_CMVN, F_FE_ASINH, T_ASINH, 0, 1);
    for (u32 r = 0; r < nf; r++) tr16(TT(K_ASINH, n_lin + r), Y80 + r * FRAME, FRAME);
    fan_rows(S_FE_LIN, Y80, FRAME, nf, HB + 4 * D, D, 0, FAN_O_S16);
    for (u32 r = 0; r < nf; r++) tr16(TT(K_LIN, n_lin + r), HB + (4 + r) * D, D);
    if (nf) v_lut(HB + 4 * D, D, nf, D, F_FE_LIN, F_FE_LIN, T_SILU, 1, 0);
    u32 lin_prev = n_lin; n_lin += nf;
    // conv1: output j reads linear rows 2j - 4 .. 2j = HB rows 2j - lin_prev .. (rows 0..3 = the previous 4)
    u32 n1 = final ? (n_lin + 1) / 2 : n_lin / 2, new1 = n1 - n_c1, c1_prev = n_c1;
    fan_rows(S_FE_C1, HB + (2 * n_c1 - lin_prev) * D, 2 * D, new1, CB + 4 * 2 * D, 2 * D, 0, FAN_O_S16);
    if (new1) v_lut(CB + 4 * 2 * D, 2 * D, new1, 2 * D, F_FE_C1, F_FE_C1, T_SILU, 1, 0);
    for (u32 r = 0; r < new1; r++) tr16(TT(K_C1, n_c1 + r), CB + (4 + r) * 2 * D, 2 * D);
    n_c1 = n1;
    if (nf >= 4) v_copy(HB + nf * D, 1, 0, HB, 1, 0, 4 * D, 1);
    // conv2: output j reads conv1 rows 2j - 4 .. 2j = CB rows 2j - c1_prev ..; into the residual ring
    u32 n2 = final ? (n_c1 + 1) / 2 : n_c1 / 2;
    while (n_c2 < n2) {
        u32 n = n2 - n_c2, s = SLOT(n_c2);
        if (n > FAN_CMAX) n = FAN_CMAX;
        if (n > RING - s) n = RING - s;
        fan(S_FE_C2, CB + (2 * n_c2 - c1_prev) * 2 * D, 4 * D, n, XR + s * D, D, XPL, FAN_O_RES);
        for (u32 c = 0; c < n; c++) tr25(TT(K_FEAT, n_c2 + c), XR + (s + c) * D, XPL, D);
        n_c2 += n;
    }
    if (new1 >= 4) v_copy(CB + new1 * 2 * D, 1, 0, CB, 1, 0, 4 * 2 * D, 1);
    nT[0] = n_c2;
    for (int i = 0; i < NLY; i++) enc_layer(i, final);
    while (n_mem < nT[NLY]) {
        u32 n = nT[NLY] - n_mem;
        if (n > FAN_CMAX) n = FAN_CMAX;
        if (n > RING - SLOT(n_mem)) n = RING - SLOT(n_mem);
        memory_rows(n_mem, n);
        n_mem += n;
    }
}

// ------------------------------------------------------------------------------------------ decoder
static int started;
static void detok(u32 id) {
    fl_seek(OFF_VOCAB + 4 * id);
    u32 a = fl_u32(), b = fl_u32();
    if (a == b) return;
    fl_seek(OFF_VOCAB + 4 * (VOCAB_N + 1) + a);
    for (; a < b; a++) {
        int c = (int)fl_u8();
        if (!started && c == ' ') { started = 1; continue; }
        started = 1;
        uart_putc(c);
    }
}
static void trd(u32 u, u32 t, u32 l, int name, u32 e, u32 n) { if (tron && t < TRACE_STEPS) tr16(TT(K_DEC + name, DI(u, t, l)), e, n); }
static void trd25(u32 u, u32 t, u32 l, int name, u32 lo) { if (tron && t < TRACE_STEPS) tr25(TT(K_DEC + name, DI(u, t, l)), lo, FAN_CMAX * D, D); }

// Decoder rows: C consecutive positions t0 .. t0 + C - 1 (C <= FAN_CMAX) with known input tokens tin[], computed
// together: every weight read of the fan-out array serves the C rows (int8 weights: the ROM port's 16 bytes per clock
// feed 32 lanes only every second clock at one row, every clock from two rows on). Each row's arithmetic is the
// same as a single step's (the array, the per-element unit and the attention work per row), so a row whose input
// token is the greedy output of the row before is bit-exact with the sequential decoder. Self-attention: all C rows'
// K and V go into the cache first; row c reads keys 0 .. t0 + c only. out[c] = argmax of row c.
#ifndef DEC_BATCH
#define DEC_BATCH 2               // rows per batch (1: one step at a time, the sequential decoder). Two rows already
                                  // keep the array's 32 lanes busy with int8 weights; more rows only add the rows
                                  // computed past a mismatch (reports/p1-core.md section 6)
#endif
#define DPL (FAN_CMAX * D)        // plane offset of the rows' 25-bit residual
static u32 dM, HDV, YV, QV, KV, VV, AV, F1, PV, LG;
static u32 PIDS[MAX_TOK], np_ids;  // the previous update's ids (none at a segment's start)
static void dec_rows(u32 u, u32 t0, u32 C, const u32 *tin, u32 *out) {
    u32 MT = dMT, MTp = dMTp, M = dM;
    for (u32 c = 0; c < C; c++) {
        u32 tok = tin[c], h = HDV + c * D;
        fl_seek(OFF_EMB_REC + 4 * tok);
        i64 em = (i64)fl_u16(); int esh = (int)fl_u8();
        fl_seek(OFF_EMB_Q + (EMB_Q4 ? D / 2 : D) * tok);
        u32 eb = 0;
        for (u32 f = 0; f < D; f++) wr25(h + f, DPL, sat(rr64(rowq(EMB_Q4, f, &eb) * em, esh), RES_SAT));
        trd25(u, t0 + c, 0, DN_EMB, h);
    }
    for (int i = 0; i < NLY; i++) {
        for (u32 c = 0; c < C; c++) { norm25(HDV + c * D, DPL, YV + c * D, N_DEC(i, 0)); trd(u, t0 + c, i, DN_LN1, YV + c * D, D); }
        fan(S_DEC(i, 0), YV, D, C, QV, D, 0, FAN_O_S16);
        fan(S_DEC(i, 1), YV, D, C, KV, D, 0, FAN_O_S16);
        fan(S_DEC(i, 2), YV, D, C, VV, D, 0, FAN_O_S16);
        for (u32 c = 0; c < C; c++) {
            u32 t = t0 + c;
            v_rope(QV + c * D, t); v_rope(KV + c * D, t);
            trd(u, t, i, DN_QROT, QV + c * D, D); trd(u, t, i, DN_KROT, KV + c * D, D); trd(u, t, i, DN_V, VV + c * D, D);
            v_copy(KV + c * D, 1, 0, KC[i] + t, MTp, 0, D, 1);
            v_copy(VV + c * D, 1, HD, VC[i] + t * HP, 1, MT * HP, HD, NH);
        }
        for (u32 c = 0; c < C; c++) {
            u32 t = t0 + c;
            for (u32 kb = 0; kb <= t; kb += B) scores(2u * (QV + c * D), KC[i], MTp, kb, SC + kb, 0);
            soft_mix(0, t + 1, FQ_D[i] + FK_D[i] + 4, VC[i], MT * HP, 0, 0, 0, 0, AV + c * D);
            trd(u, t, i, DN_ATTN, AV + c * D, D);
        }
        fan(S_DEC(i, 3), AV, D, C, HDV, D, DPL, FAN_O_RES | FAN_RESACC);
        for (u32 c = 0; c < C; c++) {
            trd25(u, t0 + c, i, DN_RES1, HDV + c * D);
            norm25(HDV + c * D, DPL, YV + c * D, N_DEC(i, 1)); trd(u, t0 + c, i, DN_LN2, YV + c * D, D);
        }
        fan(S_DEC(i, 4), YV, D, C, QV, D, 0, FAN_O_S16);
        for (u32 c = 0; c < C; c++) {
            trd(u, t0 + c, i, DN_CQ, QV + c * D, D);
            for (u32 kb = 0; kb < M; kb += B) scores(2u * (QV + c * D), CK8b[i], MAXF, kb, SC + kb, MAC_X8);
            soft_mix(0, M, FCQ[i] + FCK8[i] + 4, CV8b[i], MAXF * HP, 0, 0, FCV[i] - FCV8[i], MAC_X8, AV + c * D);
            trd(u, t0 + c, i, DN_CATTN, AV + c * D, D);
        }
        fan(S_DEC(i, 7), AV, D, C, HDV, D, DPL, FAN_O_RES | FAN_RESACC);
        for (u32 c = 0; c < C; c++) {
            trd25(u, t0 + c, i, DN_RES2, HDV + c * D);
            norm25(HDV + c * D, DPL, YV + c * D, N_DEC(i, 2)); trd(u, t0 + c, i, DN_LN3, YV + c * D, D);
        }
        fan(S_DEC(i, 8), YV, D, C, F1, 2 * FFD, 0, FAN_O_S16);
        for (u32 c = 0; c < C; c++) trd(u, t0 + c, i, DN_FC1, F1 + c * 2 * FFD, 2 * FFD);
        v_lut(F1 + FFD, FFD, C, 2 * FFD, FF_D[i], FF_D[i], T_SILU, 1, 0);
        for (u32 c = 0; c < C; c++) trd(u, t0 + c, i, DN_SILU, F1 + c * 2 * FFD + FFD, FFD);
        v_mulr(F1, 2 * FFD, FFD, PV, FFD, FFD, C, 2 * FF_D[i] - FPR_D[i]);
        for (u32 c = 0; c < C; c++) trd(u, t0 + c, i, DN_PROD, PV + c * FFD, FFD);
        fan(S_DEC(i, 9), PV, FFD, C, HDV, D, DPL, FAN_O_RES | FAN_RESACC);
        for (u32 c = 0; c < C; c++) trd25(u, t0 + c, i, DN_RES3, HDV + c * D);
    }
    for (u32 c = 0; c < C; c++) { norm25(HDV + c * D, DPL, YV + c * D, N_DECLN); trd(u, t0 + c, 0, DN_LN, YV + c * D, D); }
    for (u32 c = 0; tron && c < C; c++) if (t0 + c < TRACE_STEPS) {
        fan(S_HEAD, YV + c * D, D, 1, LG, 32768, 32768, FAN_O_RES);
        tr25(TT(K_DEC + DN_LOGITS, DI(u, t0 + c, 0)), LG, 32768, 32768);
    }
    fan(S_HEAD, YV, D, C, 0, 0, 0, FAN_O_ARGMAX);
    for (u32 c = 0; c < C; c++) out[c] = FAN->argc[c];
}

// Decode the segment against memory frames [0, M); returns the number of ids. Greedy, as IntStream: the
// output of step t is the input of step t + 1, until EOS or MT steps. L = 0: from BOS. L > 0 (a tail update): the
// first L ids are the previous update's, PIDS[0 .. L - 1], whose self-attention K/V rows 0 .. L - 1 are still in the
// cache from the refresh that decoded them (every update since kept them and wrote rows >= L only); they are sent
// again and the steps run from row L on, the first with input PIDS[L - 1]. The previous update's ids are the likely
// outputs, so the steps whose inputs they give (up to np_ids) run DEC_BATCH rows at a time; row t's argmax
// is checked against the id that was fed to row t + 1, and at the first difference the rows after it are dropped
// (their K/V cache entries are written again before anything reads them) and decoding goes on one step at a time
// from the new id. The ids, the text and every kept row's arithmetic are those of the sequential decoder.
static u32 decode(u32 u, u32 M, int final, u32 L) {
    u32 mark = brk_e, MT = (13 * M + 99) / 100, n = 0, sv[5] = {SC, SCP, EB, RCB, MIX};
    if (MT > MTC) MT = MTC;
    trace(TT(K_UPD, u), M);
    started = 0;
    if (M) {
        dM = M;
        attn_scratch(M > MT ? M : MT);
        HDV = al(2 * DPL); YV = al(DPL); QV = al(DPL); KV = al(DPL); VV = al(DPL); AV = al(DPL);
        F1 = al(FAN_CMAX * 2 * FFD); PV = al(FAN_CMAX * FFD);
        LG = tron ? al(2 * 32768) : 0;
        u32 tok = BOS, t = 0, known = DEC_BATCH > 1 ? np_ids : 0, stop = 0;
        for (; t < L; t++) { trace(TT(K_TOK, DI(u, 0, 0) | t), PIDS[t]); detok(PIDS[t]); n++; }
        if (L) tok = PIDS[L - 1];
        while (t < MT && !stop) {
            u32 C = 1, tin[FAN_CMAX], out[FAN_CMAX], c;
            if (t <= known) { C = known + 1 - t; if (C > DEC_BATCH) C = DEC_BATCH; if (C > MT - t) C = MT - t; }
            tin[0] = tok;
            for (c = 1; c < C; c++) tin[c] = PIDS[t + c - 1];
            dec_rows(u, t, C, tin, out);
            for (c = 0; c < C; c++) {
                u32 o = out[c], p = t + c;
                if (o == EOS) { stop = 1; break; }
                trace(TT(K_TOK, DI(u, 0, 0) | p), o);
                detok(o);
                n++;
                int same = p < known && o == PIDS[p];
                PIDS[p] = o; tok = o;
                if (!same) { known = 0; c++; break; }   // the rows after p were fed another id
            }
            t += c;
        }
    }
    np_ids = final ? 0 : n;
    uart_putc(final ? '\n' : '\r');
    trace(TT(K_UEND, u), n);
    brk_e = mark; SC = sv[0]; SCP = sv[1]; EB = sv[2]; RCB = sv[3]; MIX = sv[4];
    return n;
}

// ------------------------------------------------------------------------------------------ top
// One segment per capture segment. The capture buffer (rtl/top/imh_capture.sv) decides the segments: VAD-gated, by
// golden format 3.4 (opens on the VAD with up to 0.64 s of pre-roll from its ring, closes after 63 chunks when the
// VAD is off at a chunk's end, or at the 250-chunk cap), so the core sleeps between segments; rec-gated, the
// stretch while rec is high. Load guard (LG_BKL): a partial update is skipped while the capture backlog is above
// LG_BKL samples; the update at a segment's end always runs, and the final text does not depend on the partial
// ones. A tail update whose refresh was skipped is skipped too (its K/V rows were never computed). Update numbers
// count the update slots of the schedule, skipped or not, so update u of a segment is the golden model's update u.
// Events: SEG (0, the segment's first stream index), SLOT (slot, chunk count | L << 16 | refresh << 31), UPD / TOK /
// UEND per update, SKIP (slot, backlog | no-refresh << 30), SEGE (0, the stream index after its last sample), CAP
// (capture high-water mark | overrun << 31), DONE (heap high-water mark).
#ifndef LG_BKL
#define LG_BKL 8000               // samples (0.5 s); 0: never skip
#endif
static const u16 SCH_U[] = {SCHED_U}, SCH_R[] = {SCHED_R};
static u32 sch_next(const u16 *t, u32 k) { while (k >= t[0]) t += 2; return k + t[1]; }
int model_main(void) {
    tron = hal_trace_on();
    brk_e = UP(hal_heap_start() / 2); end_e = hal_heap_end() / 2 / B * B; top_e = end_e; hiw_e = 0;
    i32 st = hal_seg_begin();
    if (st < 0) return 0;
    trace(TT(K_SEG, 0), (u32)st);
    seg_init();
    np_ids = 0;
    u32 s = (u32)st, inb = 0, nch = 0, nup = 0, pend = 0, ku = sch_next(SCH_U, 0), kr = sch_next(SCH_R, 0);
    u32 lt = 0, kv_ok = 1;                            // the last refresh's kept ids, and whether it ran
    i16 v;
    while (hal_sample(&v)) {
        if (pend) {                                   // a partial update: the segment continues
            u32 b = hal_backlog(), rf = pend == 2, M = nch * 4 > 16 ? nch * 4 - 16 : 0;
            trace(TT(K_SLOT, nup), nch | (rf ? 0 : lt) << 16 | rf << 31);
            if (!rf && !kv_ok) trace(TT(K_SKIP, nup), b | 1u << 30);
            else if (LG_BKL && b > LG_BKL) { trace(TT(K_SKIP, nup), b); if (rf) kv_ok = 0; }
            else if (rf) { u32 n = decode(nup, M, 0, 0); lt = n > TAIL_KEEP ? n - TAIL_KEEP : 0; kv_ok = 1; }
            else decode(nup, M, 0, lt);
            nup++; pend = 0;
        }
        P16(CHB)[inb++] = v; s++;
        if (inb == CHUNK) {
            chunk(CHUNK / FRAME, 0); nch++; inb = 0;
            if (nch == kr) { pend = 2; kr = sch_next(SCH_R, kr); if (nch == ku) ku = sch_next(SCH_U, ku); }
            else if (nch == ku) { pend = 1; ku = sch_next(SCH_U, ku); }
        }
    }
    trace(TT(K_CAP, 0), hal_cap_info());
    u32 nf = (inb + FRAME - 1) / FRAME;
    for (u32 e = inb; e < nf * FRAME; e++) P16(CHB)[e] = 0;
    chunk(nf, 1);
    decode(nup, n_mem, 1, 0);
    trace(TT(K_SEGE, 0), s);
    trace(TT(K_DONE, 0), 2 * hiw_e);
    return 1;
}
