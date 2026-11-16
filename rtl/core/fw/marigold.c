// Marigold intent encoder on the Marigold core: the program the controller runs after every wake (docs/golden-format.md;
// reference sw/golden/intent_np.py IntIntent and sw/intent/values.py raw_values, bit for bit). From Poinsetta's
// streaming encoder (stream.c), decoder dropped.
// One segment per power-up: the island delivers its samples through the PCM port, then an end marker (PCM_END_TMO:
// no model). Every 1,280 samples (a chunk) the frontend makes 16 rows of 80 samples -> 8 conv1 rows -> 4 encoder input
// frames, and each layer releases every frame whose look-ahead exists (MG_FUT: layer 0 four frames, the others none).
// A frame that leaves the last layer goes through enc.ln into the pooled sum and the CTC value head (argmax per frame,
// repeats then blanks dropped). At the end marker the last frames are finished with the keys that exist, then the
// intent head (mean by shift, proj, gain-free norm, int8 q), the raw value of every config slot from the tokens (the
// slot types come from config SRAM), Q and VAL to the island, MATCH_GO, power off. The island picks the action,
// maps level codes, clamps the values and sends the frame.
// Weight products: the fan-out array (FAN_GO; FAN_L rows per group). Attention: the batch MAC array with RAM operands.
// Per element: the per-element unit. Storage (16-bit elements unless they say bytes): rows are frame-major; the 25-bit
// residual is two planes (low 16 bits, high 16 bits) XPL apart in a ring of XRING frames; per layer K is feature-major
// [W][KRING] (keys in the array's lanes) and V head-major [NH][VRING][HP]. Dimensions, layer count and windows come
// from layout.h (mkimage_mg.py, from the export). Peak RAM (the DONE trace, plus .bss and the 4 KB stack, of 256 KB):
// golden-format 0.3 (W 320, C1 640, FF 1280, 2 layers) 175.4 KB; draft 0.1 randtest (W 240, 6 layers) 235 KB.
#include "hal.h"
#include "layout.h"
#include "mg_values.h"
#define B LANES
#define W MG_W
#define C1 MG_C1                          // conv1 channels (conv2 reads 5 rows of C1)
#define NH MG_NH
#define HD MG_HD
#define NLY MG_NL
#define FF MG_FF
#define DQ MG_D
#define NV MG_V
#define PAST MG_PAST
#define FRAME 80
#define CHUNK 1280
#define FPC (CHUNK / (4 * FRAME))          // encoder frames per chunk (4)
#define UP(n) (((n) + B - 1) & ~(u32)(B - 1))
#define HP UP(HD)                          // V head rows padded to the lanes
#define LGS UP(NV)                         // logits row stride
// Rings. A release batch of n frames from t0 needs keys t0 - PAST .. t0 + n - 1 + fut; n <= VRING - PAST - fut keeps
// them in a V ring of VRING frames (layer 0: n <= 4; the others: n <= 8). The K ring is a multiple of the lanes so
// that the scores of every slot are computed (twice, so that a window that wraps reads straight). In flight in the
// residual ring: at most a chunk held back for the look-ahead plus a new chunk (2 * FPC frames).
#define VRING (PAST + MG_FUTMAX + FPC)
#define KRING UP(VRING)
#define XRING 16
#define MAXTOK 512
#define I16M 32767
#define I8M 127
#define P16(e) ((i16 *)(ram + 2u * (e)))
#define S_FE_LIN 0
#define S_FE_C1 1
#define S_FE_C2 2
#define S_ENC(i, j) (3 + 6 * (i) + (j))          // j: 0 q 1 k 2 v 3 o 4 fc1 5 fc2
#define N_CMVN 0
#define N_ENC(i, j) (1 + 2 * (i) + (j))          // j: 0 ln1 1 ln2
#define T_GELU 1
#define T_SILU 2
#define T_ASINH 3
#if VRING - PAST - MG_FUTMAX < 1
#error "V ring too small"
#endif
static const unsigned char FUT[NLY] = {MG_FUT};

// trace tags: kind << 20 | index (rtl/core/tools/check_mg.py has the same table)
enum { K_CMVN = 1, K_ASINH, K_LIN, K_C1, K_FEAT, K_ENC = 10, K_ELN = 20, K_CTCK, K_POOL, K_PROJ, K_Q, K_TOK, K_VAL,
       K_SEG = 40, K_END, K_T, K_NTOK, K_MATCH, K_DONE, K_CFG };
#define TT(k, i) (((u32)(k) << 20) | (u32)(i))
#define HMUL 16777619u
static int tron;

// ------------------------------------------------------------------------------------------ small helpers
static inline i64 rr64(i64 x, int s) { return s <= 0 ? x << (-s) : (x + ((i64)1 << (s - 1))) >> s; }
static inline i32 sat(i64 v, i32 m) { return v > m ? m : v < -m ? -m : (i32)v; }
static u32 brk_e, end_e, hiw_e;
static u32 al(u32 n) { u32 e = brk_e; brk_e += UP(n); if (brk_e > hiw_e) hiw_e = brk_e; if (brk_e > end_e) hal_fail(1); return e; }
static inline i32 rd25(u32 lo, u32 pl) { return (i32)(((u32)(u16)P16(lo)[pl] << 16) | (u16)P16(lo)[0]); }
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
static u32 cfg_u8(u32 off) { return (mg_cfg32(off & ~3u) >> (8 * (off & 3))) & 255u; }
static u32 cfg_u16(u32 off) { return cfg_u8(off) | (cfg_u8(off + 1) << 8); }

// ------------------------------------------------------------------------------------------ per-element unit
static void v_copy(u32 src, u32 sstr, u32 srow, u32 dst, u32 dstr, u32 drow, u32 n, u32 rows) {
    VEC->src = src; VEC->sstr = sstr; VEC->srow = srow; VEC->dst = dst; VEC->dstr = dstr; VEC->drow = drow;
    VEC->n = n; VEC->rows = rows;
    VEC_GO(VEC_COPY);
}
static void v_lut(u32 a, u32 n, u32 rows, u32 row, int f, int fo, int tab, int tails, int r4) {
    VEC->src = a; VEC->sstr = 1; VEC->srow = row; VEC->dst = a; VEC->dstr = 1; VEC->drow = row; VEC->n = n; VEC->rows = rows;
    VEC->p0 = (u32)f | ((u32)fo << 4) | ((u32)tab << 8) | ((u32)tails << 10) | ((u32)r4 << 13);
    VEC_GO(VEC_LUT);
}
static void v_norm(u32 src, u32 pl, int wide, u32 dst, u32 n, int idx) {
    VEC->p0 = n; VEC->p1 = NORMS[idx].eps;
    VEC_GO(VEC_NSTAT);
    VEC->src = src; VEC->sstr = 1; VEC->srow = 0; VEC->dst = dst; VEC->dstr = 1; VEC->drow = 0;
    VEC->n = n; VEC->rows = 1; VEC->p1 = pl;
    VEC->p3 = (u32)(12 + NORMS[idx].fg - NORMS[idx].fout) << 8; VEC->faddr = NORMS[idx].off;
    VEC_GO(VEC_NORM | VEC_AUTO | (wide ? VEC_WIDE : 0));
}
// LayerNorm of a 25-bit row (planes lo, lo + pl)
static void norm25(u32 lo, u32 pl, u32 dst, int idx) {
    VEC->src = lo; VEC->sstr = 1; VEC->n = W; VEC->rows = 1; VEC->p1 = pl;
    VEC_GO(VEC_STAT | VEC_CLR | VEC_WIDE);
    v_norm(lo, pl, 1, dst, W, idx);
}
// norm of a 16-bit row of n values
static void norm16(u32 src, u32 n, u32 dst, int idx) {
    VEC->src = src; VEC->sstr = 1; VEC->n = n; VEC->rows = 1;
    VEC_GO(VEC_STAT | VEC_CLR);
    v_norm(src, 0, 0, dst, n, idx);
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

// ------------------------------------------------------------------------------------------ segment state
static u32 CHB, Y80, HB, CB, XR, XPL, KR[NLY], VR[NLY], TA, TB, TC, QB, SC, SCP, EB, RCB, MIX;
static u32 n_lin, n_c1, n_c2, nT[NLY + 1], kvn[NLY], done[NLY], n_out;
static i64 pool[W];
static u8 tok[MAXTOK];
static u32 ntok;
static i32 prevk;
#define SX(t) ((t) % XRING)
#define SK(t) ((t) % KRING)
#define SV(t) ((t) % VRING)

static void seg_init(void) {
    CHB = al(CHUNK); Y80 = al(16 * FRAME);
    HB = al(20 * W); CB = al(12 * C1);
    XPL = XRING * W; XR = al(2 * XPL);
    for (int i = 0; i < NLY; i++) { KR[i] = al(W * KRING); VR[i] = al(NH * VRING * HP); }
    TA = al(FAN_CMAX * W); TB = al(FAN_CMAX * W); TC = al(FAN_CMAX * (FF > LGS ? FF : LGS)); QB = al(FAN_CMAX * W);
    SCP = 2 * KRING; SC = al(3 * NH * SCP); EB = al(2 * NH * KRING); RCB = al(2 * NH); MIX = al(NH * HP);
    for (u32 e = 0; e < 4 * W; e++) P16(HB)[e] = 0;                      // the convolutions' causal padding
    for (u32 e = 0; e < 4 * C1; e++) P16(CB)[e] = 0;
    for (int i = 0; i < NLY; i++)                                          // V rows' pad lanes (read, never used)
        for (u32 r = 0; r < NH * VRING; r++) for (u32 d = HD; d < HP; d++) P16(VR[i])[r * HP + d] = 0;
    n_lin = n_c1 = n_c2 = n_out = 0;
    for (int i = 0; i <= NLY; i++) nT[i] = 0;
    for (int i = 0; i < NLY; i++) { kvn[i] = 0; done[i] = 0; }
    for (u32 c = 0; c < W; c++) pool[c] = 0;
    ntok = 0; prevk = -1;
}

// ------------------------------------------------------------------------------------------ attention
static void scores(u32 wq, u32 K, u32 kb, u32 y) {
    MAC->w = wq; MAC->wi = 2; MAC->wo = 2 * HD; MAC->xi = KRING; MAC->xo = HD * KRING; MAC->xw = 0;
    MAC->nin = HD; MAC->nout = NH; MAC->ys = SCP; MAC->yp = NH * SCP; MAC->x = K + kb; MAC->y = y;
    MAC_GO(MAC_W_RAM16 | MAC_O_RAW);
}
// band attention of layer i for the query row at qa over keys [j0, j0 + nk): softmax of the raw scores (window
// starting at K slot SK(j0)), then mixing with the V ring (key i at V slot (SV(j0) + i) mod VRING)
static void band_attn(int i, u32 qa, u32 j0, u32 nk, u32 out) {
    for (u32 m = 0; m < 2; m++)
        for (u32 kb = 0; kb < KRING; kb += B) scores(2u * qa, KR[i], kb, SC + m * KRING + kb);
    VEC->src = SC + SK(j0); VEC->sstr = 1; VEC->srow = SCP; VEC->p1 = NH * SCP; VEC->n = nk; VEC->rows = NH;
    VEC->dst = EB; VEC->dstr = 2; VEC->drow = 2 * nk; VEC->p0 = (u32)SM_C | ((u32)(FQ_E[i] + FK_E[i] + 4) << 16);
    VEC->p2 = RCB;
    VEC_GO(VEC_SOFTMAX);
    MAC->w = 2u * EB; MAC->wi = 4; MAC->wo = 4 * nk; MAC->xi = HP; MAC->xo = VRING * HP; MAC->xw = VRING * HP;
    MAC->xs0 = SV(j0) * HP; MAC->nin = nk; MAC->nout = NH; MAC->ys = HP; MAC->yp = 0; MAC->h = 2u * RCB; MAC->hsh = 44;
    for (u32 g = 0; g < HP; g += B) { MAC->x = VR[i] + g; MAC->y = MIX + g; MAC_GO(MAC_W_RAM32 | MAC_O_S16 | MAC_HDRRAM); }
    MAC->xw = 0; MAC->xs0 = 0;
    v_copy(MIX, 1, HP, out, 1, HD, HD, NH);
}

// ------------------------------------------------------------------------------------------ encoder
// K and V of layer i's input frames t .. t + n - 1 (ln1 rows in TA) into the rings
static void kv_store(int i, u32 t, u32 n) {
    fan(S_ENC(i, 1), TA, W, n, TB, W, 0, FAN_O_S16);
    fan(S_ENC(i, 2), TA, W, n, TC, W, 0, FAN_O_S16);
    for (u32 c = 0; c < n; c++) {
        v_copy(TB + c * W, 1, 0, KR[i] + SK(t + c), KRING, 0, W, 1);
        v_copy(TC + c * W, 1, HD, VR[i] + SV(t + c) * HP, 1, VRING * HP, HD, NH);
    }
}
static void ln1_rows(int i, u32 t, u32 n) {
    for (u32 c = 0; c < n; c++) norm25(XR + SX(t + c) * W, XPL, TA + c * W, N_ENC(i, 0));
}

// layer i: release every frame whose look-ahead exists (all of them when final), in batches; the keys a batch needs
// are computed just before it. A layer without look-ahead computes K, V and q of a batch in one pass.
static void enc_layer(int i, int final) {
    u32 T = nT[i], f = FUT[i];
    u32 ready = final ? T : (T > f ? T - f : 0);
    while (done[i] < ready) {
        u32 t0 = done[i], n = ready - t0, sx = SX(t0);
        if (n > FAN_CMAX) n = FAN_CMAX;
        if (n > XRING - sx) n = XRING - sx;
        if (n > VRING - PAST - f) n = VRING - PAST - f;
        u32 need = (t0 + n - 1 + f < T ? t0 + n - 1 + f : T - 1) + 1;
        if (kvn[i] == t0 && need == t0 + n) {
            ln1_rows(i, t0, n);
            fan(S_ENC(i, 0), TA, W, n, QB, W, 0, FAN_O_S16);
            kv_store(i, t0, n);
            kvn[i] = need;
        } else {
            while (kvn[i] < need) {                 // keys ahead of the batch, in runs that do not wrap the ring
                u32 t = kvn[i], m = need - t;
                if (m > FAN_CMAX) m = FAN_CMAX;
                if (m > XRING - SX(t)) m = XRING - SX(t);
                ln1_rows(i, t, m);
                kv_store(i, t, m);
                kvn[i] += m;
            }
            ln1_rows(i, t0, n);
            fan(S_ENC(i, 0), TA, W, n, QB, W, 0, FAN_O_S16);
        }
        for (u32 c = 0; c < n; c++) {
            u32 t = t0 + c, j0 = t > PAST ? t - PAST : 0, j1 = t + f < T ? t + f : T - 1;
            band_attn(i, QB + c * W, j0, j1 - j0 + 1, TA + c * W);
        }
        u32 xr = XR + sx * W;
        fan(S_ENC(i, 3), TA, W, n, xr, W, XPL, FAN_O_RES | FAN_RESACC);
        for (u32 c = 0; c < n; c++) norm25(xr + c * W, XPL, TB + c * W, N_ENC(i, 1));
        fan(S_ENC(i, 4), TB, W, n, TC, FF, 0, FAN_O_S16);
        v_lut(TC, FF, n, FF, FF_E[i], FF_E[i], T_GELU, 1, 0);
        fan(S_ENC(i, 5), TC, FF, n, xr, W, XPL, FAN_O_RES | FAN_RESACC);
        for (u32 c = 0; c < n; c++) tr25(TT(K_ENC + i, t0 + c), xr + c * W, XPL, W);
        done[i] += n;
    }
    nT[i + 1] = done[i];
}

// frames out of the last layer: enc.ln, the pooled sum, CTC logits -> argmax -> tokens (repeats, then blanks dropped)
static void out_frames(void) {
    while (n_out < nT[NLY]) {
        u32 t0 = n_out, n = nT[NLY] - t0, sx = SX(t0);
        if (n > FAN_CMAX) n = FAN_CMAX;
        if (n > XRING - sx) n = XRING - sx;
        for (u32 c = 0; c < n; c++) {
            norm25(XR + (sx + c) * W, XPL, TB + c * W, N_ENCLN);
            tr16(TT(K_ELN, t0 + c), TB + c * W, W);
            const i16 *e = P16(TB + c * W);
            for (u32 ch = 0; ch < W; ch++) pool[ch] += e[ch];
        }
        fan(S_CTC, TB, W, n, TC, LGS, 0, FAN_O_S16);
        for (u32 c = 0; c < n; c++) {
            const i16 *lg = P16(TC + c * LGS);
            u32 k = 0;
            for (u32 o = 1; o < NV; o++) if (lg[o] > lg[k]) k = o;      // first maximum
            if (tron) trace(TT(K_CTCK, t0 + c), k);
            if ((i32)k != prevk && k != 0) { if (ntok < MAXTOK) tok[ntok] = (u8)k; ntok++; }
            prevk = (i32)k;
        }
        n_out += n;
    }
}

// one chunk of nf frames of 80 samples in CHB (final: the segment's last, zero-padded)
static void chunk(u32 nf, int final) {
    for (u32 r = 0; r < nf; r++) {
        norm16(CHB + r * FRAME, FRAME, Y80 + r * FRAME, N_CMVN);
        tr16(TT(K_CMVN, n_lin + r), Y80 + r * FRAME, FRAME);
    }
    if (nf) v_lut(Y80, FRAME, nf, FRAME, F_FE_CMVN, F_FE_ASINH, T_ASINH, 0, 1);
    for (u32 r = 0; r < nf; r++) tr16(TT(K_ASINH, n_lin + r), Y80 + r * FRAME, FRAME);
    fan_rows(S_FE_LIN, Y80, FRAME, nf, HB + 4 * W, W, 0, FAN_O_S16);
    for (u32 r = 0; r < nf; r++) tr16(TT(K_LIN, n_lin + r), HB + (4 + r) * W, W);
    if (nf) v_lut(HB + 4 * W, W, nf, W, F_FE_LIN, F_FE_LIN, T_SILU, 1, 0);
    u32 lin_prev = n_lin; n_lin += nf;
    // conv1: output j reads linear rows 2j - 4 .. 2j = HB rows 2j - lin_prev .. (rows 0..3 = the previous 4)
    u32 n1 = final ? (n_lin + 1) / 2 : n_lin / 2, new1 = n1 - n_c1, c1_prev = n_c1;
    fan_rows(S_FE_C1, HB + (2 * n_c1 - lin_prev) * W, 2 * W, new1, CB + 4 * C1, C1, 0, FAN_O_S16);
    for (u32 r = 0; r < new1; r++) tr16(TT(K_C1, n_c1 + r), CB + (4 + r) * C1, C1);
    if (new1) v_lut(CB + 4 * C1, C1, new1, C1, F_FE_C1, F_FE_C1, T_SILU, 1, 0);
    n_c1 = n1;
    if (nf >= 4) v_copy(HB + nf * W, 1, 0, HB, 1, 0, 4 * W, 1);
    // conv2: output j reads conv1 rows 2j - 4 .. 2j = CB rows 2j - c1_prev ..; into the residual ring
    u32 n2 = final ? (n_c1 + 1) / 2 : n_c1 / 2;
    if (n2 - n_out > XRING) hal_fail(2);
    while (n_c2 < n2) {
        u32 n = n2 - n_c2, sx = SX(n_c2);
        if (n > FAN_CMAX) n = FAN_CMAX;
        if (n > XRING - sx) n = XRING - sx;
        fan(S_FE_C2, CB + (2 * n_c2 - c1_prev) * C1, 2 * C1, n, XR + sx * W, W, XPL, FAN_O_RES);
        for (u32 c = 0; c < n; c++) tr25(TT(K_FEAT, n_c2 + c), XR + (sx + c) * W, XPL, W);
        n_c2 += n;
    }
    if (new1 >= 4) v_copy(CB + new1 * C1, 1, 0, CB, 1, 0, 4 * C1, 1);
    nT[0] = n_c2;
    for (int i = 0; i < NLY; i++) enc_layer(i, final);
    out_frames();
}

// ------------------------------------------------------------------------------------------ head, values, matcher
static void head_and_match(void) {
    u32 T = n_out;
    int b = 0;
    for (u32 x = T > 0 ? T - 1 : 0; x; x >>= 1) b++;               // bit length of T - 1
    trace(TT(K_T, 0), T);
    for (u32 c = 0; c < W; c++) {
        i32 m = sat(rr64(pool[c], b), I16M);
        P16(TA)[c] = (i16)m;
        if (tron) trace(TT(K_POOL, c), (u32)m);
    }
    fan(S_PROJ, TA, W, 1, TB, UP(DQ), 0, FAN_O_S16);
    if (tron) for (u32 d = 0; d < DQ; d++) trace(TT(K_PROJ, d), (u32)(i32)P16(TB)[d]);
    norm16(TB, DQ, TC, N_OUT);
    // value head: the raw value of every config slot (types from the slot table)
    u32 D = cfg_u16(0x06), N = cfg_u16(0x08), na = cfg_u8(0x0A), ns = cfg_u8(0x0B);
    if (ns > 8) ns = 8;
    u32 so = 0x20 + 4 * N + N * D + 4 * na;
    u8 types[8]; i32 raw[8];
    for (u32 k = 0; k < ns; k++) types[k] = (u8)cfg_u8(so + 12 * k);
    u32 nt = ntok < MAXTOK ? ntok : MAXTOK;
    trace(TT(K_CFG, 0), D | (ns << 16)); trace(TT(K_NTOK, 0), ntok);
    if (tron) for (u32 j = 0; j < nt; j++) trace(TT(K_TOK, j), tok[j]);
    mg_raw_values(tok, nt, types, ns, raw, ram + 2u * KR[0]);      // scratch: the encoder's rings are free now
    // Q (four int8 per word), VAL[0..7], then MATCH_GO
    for (u32 w = 0; w < (DQ + 3) / 4; w++) {
        u32 v = 0;
        for (u32 j = 0; j < 4 && 4 * w + j < DQ; j++) {
            i32 q = sat(P16(TC)[4 * w + j], I8M);
            if (tron) trace(TT(K_Q, 4 * w + j), (u32)q);
            v |= ((u32)q & 255u) << (8 * j);
        }
        mg_isl_wr(ISL_Q + 4 * w, v);
    }
    for (u32 k = 0; k < 8; k++) {
        i32 v = k < ns ? raw[k] : 0;
        if (tron) trace(TT(K_VAL, k), (u32)v);
        mg_isl_wr(ISL_VAL + 4 * k, (u32)v);
    }
    mg_isl_wr(ISL_MATCH_GO, 1);
    if (tron) {                                                       // debug runs: the matcher's answer
        u32 st = 0;
        for (u32 k = 0; k < 20000 && !((st = mg_isl_rd(ISL_MATCH_GO)) & 2); k++) ;
        trace(TT(K_MATCH, 0), st); trace(TT(K_MATCH, 1), mg_isl_rd(ISL_MATCH_BEST));
    }
}

// ------------------------------------------------------------------------------------------ top
// Events: SEG (samples in the segment), END (end marker bits), T (encoder frames), CFG (config D | n_slots << 16),
// NTOK, MATCH (debug: MATCH_GO status, MATCH_BEST), DONE (peak RAM bytes: .bss and heap); the per-row traces and
// POOL / PROJ / Q / TOK / VAL values only with the trace strap.
int model_main(void) {
    tron = hal_trace_on();
    brk_e = UP(hal_heap_start() / 2); end_e = hal_heap_end() / 2 / B * B; hiw_e = brk_e;
    seg_init();
    u32 inb = 0, ns = 0;
    i16 v;
    while (mg_sample(&v)) {
        P16(CHB)[inb++] = v; ns++;
        if (inb == CHUNK) { chunk(CHUNK / FRAME, 0); inb = 0; }
    }
    u32 endf = mg_seg_end();
    trace(TT(K_SEG, 0), ns); trace(TT(K_END, 0), endf);
    if (!(endf & PCM_END_TMO)) {
        u32 nf = (inb + FRAME - 1) / FRAME;
        for (u32 e = inb; e < nf * FRAME; e++) P16(CHB)[e] = 0;
        chunk(nf, 1);
        head_and_match();
    }
    trace(TT(K_DONE, 0), 2 * hiw_e);
    mg_power_off();
    return 0;
}
