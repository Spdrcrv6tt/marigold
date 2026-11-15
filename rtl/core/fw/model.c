// Moonshine Tiny on the Imhotep core: the program the controller runs.
// Arithmetic follows docs/golden-format.md (sw/golden/moonshine_np.py IntModel) bit for bit.
// Per weight work is a MAC array operation (MAC_GO); per element work is an operation of the per-element unit
// (VEC_GO) or, with -DUSE_VEC=0, the same thing as C loops on the controller (slow; kept as the reference of
// what the unit replaces). What is left here is sequencing, one square root and two divisions per norm row,
// rotary positions, the SiLU gate product and the embedding lookup.
//
// Storage conventions (B = LANES, addresses in 16-bit elements from the start of the activation RAM):
//   FT tensor  [feature][frame], frames padded to a multiple of B: element (f, t) at base + f*Tp + t.
//              One RAM word then holds B consecutive frames of one feature = the B lanes of the array.
//   VH tensor  [head][key][HP] with HP = 36 rounded up to B: the layout attention mixing needs.
//   vectors    plain int16 arrays (decoder, fan-in mode).
//   residual   25 bit, as two FT planes (low 16 bits, high 16 bits) so that the array adds into it directly.
#include "hal.h"
#include "layout.h"

#ifndef USE_VEC
#define USE_VEC 1
#endif
#define B LANES
#define NLY 6
#define D 288
#define NH 8
#define HD 36
#define HP (((HD + B - 1) / B) * B)
#define I16M 32767
#define RES_SAT ((1 << 24) - 1)
#define BOS 1
#define EOS 2
#ifndef TRACE_STEPS
#define TRACE_STEPS 1000          // decoder steps whose internals are traced (all of them)
#endif
#define UP(n) (((n) + B - 1) & ~(u32)(B - 1))
#define P16(e) ((i16 *)(ram + 2u * (e)))

// site / norm indices (order of mkimage.py)
#define S_ENC(i, j) (3 + 6 * (i) + (j))          // j: 0 q 1 k 2 v 3 o 4 fc1 5 fc2
#define S_DEC(i, j) (39 + 10 * (i) + (j))        // j: 0 q 1 k 2 v 3 o 4 cq 5 ck 6 cv 7 co 8 fc1 9 fc2
#define S_EMB 99
#define N_GN 0
#define N_ENC 1
#define N_DEC 2
#define N_ENCL(i, j) (3 + 5 * (i) + (j))         // j: 0 ln1 1 ln2
#define N_DECL(i, j) (3 + 5 * (i) + 2 + (j))     // j: 0 ln1 1 ln2 2 ln3
#define T_TANH 0
#define T_GELU 1
#define T_SILU 2

// trace tags: kind << 16 | layer << 8 | step   (rtl/core/tools/events.py has the same table)
enum { K_PCM = 1, K_CONV1, K_TANH, K_GN, K_CONV2, K_GELU2, K_CONV3, K_STEM,
       K_E_LN1 = 10, K_E_Q, K_E_K, K_E_V, K_E_ATTN, K_E_RES1, K_E_LN2, K_E_FC1, K_E_GELU, K_E_RES2,
       K_ENC = 20, K_CK, K_CV,
       K_D_EMB = 30, K_D_LN1, K_D_Q, K_D_K, K_D_V, K_D_ATTN, K_D_RES1, K_D_LN2, K_D_CQ, K_D_CATTN, K_D_RES2,
       K_D_LN3, K_D_FC1, K_D_SILU, K_D_PROD, K_D_RES3, K_D_LN, K_D_ARGIDX, K_D_ARGVAL,
       K_TOKEN = 50, K_START = 59, K_NSAMP = 60, K_PHASE, K_MEM, K_DONE, K_SAT, K_CAP };
#define TAG(k, l, s) (((u32)(k) << 16) | ((u32)(l) << 8) | (u32)(s))

static int tron;
static u32 brk_e, top_e, end_e, hiw_e;   // allocators in elements: bottom-up (brk_e) and top-down (top_e);
                                         // hiw_e = high-water of brk_e + (end_e - top_e)
static u32 N, T1, T2, T, T1p, T2p, Tp, MT, MTp;

// ------------------------------------------------------------------------------------------ small helpers
static inline i64 rr64(i64 x, int s) { return s <= 0 ? x << (-s) : (x + ((i64)1 << (s - 1))) >> s; }
static inline i32 rr32(i32 x, int s) { return s <= 0 ? x << (-s) : (x + ((i32)1 << (s - 1))) >> s; }
static inline i32 sat(i64 v, i32 m) { return v > m ? m : v < -m ? -m : (i32)v; }

static u64 udiv64(u64 n, u64 d) {
    u64 q = 0, r = 0;
    for (int i = 63; i >= 0; i--) {
        r = (r << 1) | ((n >> i) & 1);
        if (r >= d) { r -= d; q |= (u64)1 << i; }
    }
    return q;
}
static i64 fdiv64(i64 n, u64 d) {           // floor(n / d), d > 0
    if (n >= 0) return (i64)udiv64((u64)n, d);
    return -(i64)udiv64((u64)(-n) + d - 1, d);
}
static u64 isqrt64(u64 n) {                 // floor(sqrt(n)), n < 2^62
    u64 res = 0, bit = (u64)1 << 60;
    for (int i = 0; i < 31; i++) {
        u64 t = res + bit;
        if (n >= t) { n -= t; res = (res >> 1) + bit; } else res >>= 1;
        bit >>= 2;
    }
    return res;
}
static int bitlen64(u64 v) { int r = 0; while (v) { r++; v >>= 1; } return r; }

static void use(void) {
    u32 u = brk_e + (end_e - top_e);
    if (u > hiw_e) hiw_e = u;
    if (brk_e > top_e) hal_fail(1);
}
static u32 al(u32 nelem) { u32 e = brk_e; brk_e += UP(nelem); use(); return e; }        // bottom, short-lived
static u32 alt(u32 nelem) { top_e -= UP(nelem); use(); return top_e; }                 // top, long-lived

static u32 fl_u16(void) { u32 a = fl_u8(); return a | (fl_u8() << 8); }
static u32 fl_u32(void) { u32 a = fl_u16(); return a | (fl_u16() << 16); }
static inline i32 rd25(u32 lo, u32 pl) { return (i32)(((u32)(u16)P16(lo)[pl] << 16) | (u16)P16(lo)[0]); }
static inline void wr25(u32 lo, u32 pl, i32 v) { P16(lo)[0] = (i16)v; P16(lo)[pl] = (i16)(v >> 16); }

// ------------------------------------------------------------------------------------------ trace
#define HMUL 16777619u
static void tr_ft(u32 tag, u32 base, u32 nf, u32 tp, u32 t, int frame_major) {
    if (!tron) return;
    u32 h = 0; const i16 *p = P16(base);
    if (frame_major) { for (u32 j = 0; j < t; j++) for (u32 f = 0; f < nf; f++) h = h * HMUL + (u32)(i32)p[f * tp + j]; }
    else             { for (u32 f = 0; f < nf; f++) for (u32 j = 0; j < t; j++) h = h * HMUL + (u32)(i32)p[f * tp + j]; }
    trace(tag, h);
}
static void tr_v16(u32 tag, const i16 *p, u32 n) {
    if (!tron) return;
    u32 h = 0; for (u32 i = 0; i < n; i++) h = h * HMUL + (u32)(i32)p[i];
    trace(tag, h);
}
// residual planes, frame-major order
static void tr_res(u32 tag, u32 lo, u32 pl, u32 tp, u32 t) {
    if (!tron) return;
    u32 h = 0;
    for (u32 j = 0; j < t; j++) for (u32 f = 0; f < D; f++) h = h * HMUL + (u32)rd25(lo + f * tp + j, pl);
    trace(tag, h);
}

// ------------------------------------------------------------------------------------------ per-element operations
typedef struct { i64 mean; i64 rec; int sh1; int sh2; } nst_t;
static void norm_scale(nst_t *st, u64 ss, u32 n, int idx) {
    u64 var = udiv64(ss, n) + (u64)NORMS[idx].eps;
    if (var < 1) var = 1;
    int s = ((61 - bitlen64(var)) >> 1) * 2;
    u64 sd = isqrt64(s >= 0 ? var << s : var >> (-s));
    st->rec = (i64)udiv64((u64)1 << 61, sd);
    st->sh1 = 61 - s / 2 - 12;
    st->sh2 = 12 + NORMS[idx].fg - NORMS[idx].fout;
}
// statistics from the sum and the sum of squares: sum((x - mean)^2) = Q - 2*mean*S + n*mean^2, exactly
static void norm_stats(nst_t *st, i64 s, u64 q, u32 n, int idx) {
    st->mean = fdiv64(2 * s + n, 2 * (u64)n);
    norm_scale(st, q - (u64)(2 * st->mean * s) + (u64)n * (u64)(st->mean * st->mean), n, idx);
}

#if USE_VEC
static void v_copy(u32 src, u32 sstr, u32 srow, u32 dst, u32 dstr, u32 drow, u32 n, u32 rows) {
    VEC->src = src; VEC->sstr = sstr; VEC->srow = srow; VEC->dst = dst; VEC->dstr = dstr; VEC->drow = drow;
    VEC->n = n; VEC->rows = rows;
    VEC_GO(VEC_COPY);
}
// in place over a 2-D region; returns sum and sum of squares of the results through S/Q when acc is set
static void v_lut(u32 a, u32 str, u32 row, u32 n, u32 rows, int f, int fo, int tab, int tails, int sa, int acc) {
    VEC->src = a; VEC->sstr = str; VEC->srow = row; VEC->dst = a; VEC->dstr = str; VEC->drow = row;
    VEC->n = n; VEC->rows = rows;
    VEC->p0 = (u32)f | ((u32)fo << 4) | ((u32)tab << 8) | ((u32)tails << 10) | ((u32)sa << 11) | ((u32)acc << 12);
    VEC_GO(VEC_LUT | (acc ? VEC_CLR : 0));
}
static void v_sq(i64 *s, u64 *q) {
    *s = (i64)(((u64)VEC->s_hi << 32) | VEC->s_lo); *q = ((u64)VEC->q_hi << 32) | VEC->q_lo;
}
// normalise a 2-D region with the statistics the unit derives from S and Q (cnt values); G/B records come
// from the weight source
static void v_norm(u32 src, u32 pl, u32 sstr, u32 srow, int wide, u32 dst, u32 dstr, u32 drow, u32 n, u32 rows,
                   u32 cnt, int idx, int gbrow) {
    VEC->p0 = cnt; VEC->p1 = (u32)NORMS[idx].eps;
    VEC_GO(VEC_NSTAT);
    VEC->src = src; VEC->sstr = sstr; VEC->srow = srow; VEC->dst = dst; VEC->dstr = dstr; VEC->drow = drow;
    VEC->n = n; VEC->rows = rows; VEC->p1 = pl;
    VEC->p3 = (u32)(12 + NORMS[idx].fg - NORMS[idx].fout) << 8; VEC->faddr = NORMS[idx].off;
    VEC_GO(VEC_NORM | VEC_AUTO | (wide ? VEC_WIDE : 0) | (gbrow ? VEC_GBROW : 0));
}
// LayerNorm of 288 channels: x 25 bit in planes (lo, lo + pl) with stride sstr
static void norm_row(u32 lo, u32 pl, u32 sstr, u32 dst, u32 dstr, int idx) {
    VEC->src = lo; VEC->sstr = sstr; VEC->n = D; VEC->rows = 1; VEC->p1 = pl;
    VEC_GO(VEC_STAT | VEC_CLR | VEC_WIDE);
    v_norm(lo, pl, sstr, 0, 1, dst, dstr, 0, D, 1, D, idx, 0);
}
// rotary position pos applied to a 288-vector with the given element stride
static void v_rope(u32 v, u32 stride, u32 pos) {
    VEC->src = v; VEC->sstr = 2 * stride; VEC->srow = HD * stride; VEC->p1 = stride; VEC->n = 16; VEC->rows = NH;
    VEC->faddr = OFF_ROPE + 64 * pos;
    VEC_GO(VEC_ROPE);
}
static void v_mul(u32 a, u32 boff, u32 dst, u32 n, int sh) {
    VEC->src = a; VEC->sstr = 1; VEC->srow = 0; VEC->dst = dst; VEC->dstr = 1; VEC->drow = 0; VEC->n = n; VEC->rows = 1;
    VEC->p0 = (u32)sh; VEC->p1 = boff;
    VEC_GO(VEC_MUL);
}
#else
// ---- the same operations as C on the controller
static i16 NG[D];
static i32 NB[D];
static int norm_loaded;
static i64 sw_s; static u64 sw_q;
static void v_copy(u32 src, u32 sstr, u32 srow, u32 dst, u32 dstr, u32 drow, u32 n, u32 rows) {
    for (u32 r = 0; r < rows; r++) for (u32 i = 0; i < n; i++) P16(dst)[r * drow + i * dstr] = P16(src)[r * srow + i * sstr];
}
static i32 lut(i32 x, int f, const int *tab, int tails, int f_out) {
    int fw = f > 5 ? f : 5;
    x <<= (fw - f);
    i32 lim = (i32)8 << fw;
    i32 u = x < -lim ? -lim : x > lim - 1 ? lim - 1 : x;
    u += lim;
    int sh = fw - 5;
    i32 idx = u >> sh, fr = u & ((1 << sh) - 1);
    i32 y = rr32(tab[idx] + (((tab[idx + 1] - tab[idx]) * fr) >> sh), 14 - f_out);
    if (tails) {
        if (x >= lim) y = rr32(x, fw - f_out);
        if (x < -lim) y = 0;
    }
    return y;
}
static void v_lut(u32 a, u32 str, u32 row, u32 n, u32 rows, int f, int fo, int tab, int tails, int sa, int acc) {
    const int *T = tab == T_TANH ? LUT_TANH : tab == T_GELU ? LUT_GELU : LUT_SILU;
    if (acc) { sw_s = 0; sw_q = 0; }
    for (u32 r = 0; r < rows; r++) for (u32 i = 0; i < n; i++) {
        i16 *p = P16(a) + r * row + i * str;
        i32 y = lut(*p, f, T, tails, fo);
        if (sa) y = sat(y, I16M);
        *p = (i16)y;
        if (acc) { sw_s += y; sw_q += (u64)((i64)y * y); }
    }
}
static void v_sq(i64 *s, u64 *q) { *s = sw_s; *q = sw_q; }
static void norm_load(int idx) {
    if (norm_loaded == idx + 1) return;
    fl_seek(NORMS[idx].off);
    for (int i = 0; i < D; i++) { NG[i] = (i16)fl_u16(); NB[i] = (i32)fl_u32(); }
    norm_loaded = idx + 1;
}
static inline i16 norm_el(const nst_t *st, i32 x, int c) {
    i32 y = (i32)rr64((i64)(x - (i32)st->mean) * st->rec, st->sh1);
    return (i16)sat(rr64((i64)y * NG[c], st->sh2) + NB[c], I16M);
}
static void v_norm(u32 src, u32 pl, u32 sstr, u32 srow, int wide, u32 dst, u32 dstr, u32 drow, u32 n, u32 rows,
                   u32 cnt, int idx, int gbrow) {
    nst_t st0; const nst_t *st = &st0;
    norm_stats(&st0, sw_s, sw_q, cnt, idx);
    norm_load(idx);
    for (u32 r = 0; r < rows; r++) for (u32 i = 0; i < n; i++) {
        u32 e = src + r * srow + i * sstr;
        i32 x = wide ? rd25(e, pl) : P16(e)[0];
        P16(dst)[r * drow + i * dstr] = norm_el(st, x, gbrow ? r : i);
    }
}
static void norm_row(u32 lo, u32 pl, u32 sstr, u32 dst, u32 dstr, int idx) {
    sw_s = 0; sw_q = 0;
    for (int c = 0; c < D; c++) { i64 x = rd25(lo + c * sstr, pl); sw_s += x; sw_q += (u64)(x * x); }
    v_norm(lo, pl, sstr, 0, 1, dst, dstr, 0, D, 1, D, idx, 0);
}
static void rope_vec(i16 *v, u32 stride, const i16 *cs) {
    for (int h = 0; h < NH; h++)
        for (int j = 0; j < 16; j++) {
            i16 *p1 = v + (h * HD + 2 * j) * stride, *p2 = p1 + stride;
            i32 x1 = *p1, x2 = *p2, c = cs[j], s = cs[16 + j];
            *p1 = (i16)sat(rr32(x1 * c - x2 * s, 14), I16M);
            *p2 = (i16)sat(rr32(x2 * c + x1 * s, 14), I16M);
        }
}
static void v_rope(u32 v, u32 stride, u32 pos) {
    i16 cs[32];
    fl_seek(OFF_ROPE + 64 * pos);
    for (int j = 0; j < 32; j++) cs[j] = (i16)fl_u16();
    rope_vec(P16(v), stride, cs);
}
static void v_mul(u32 a, u32 boff, u32 dst, u32 n, int sh) {
    for (u32 j = 0; j < n; j++) P16(dst)[j] = (i16)sat(rr64((i64)P16(a)[j] * P16(a)[boff + j], sh), I16M);
}
#endif

// ------------------------------------------------------------------------------------------ MAC array calls
static void mac_site(int s, u32 x, u32 xi, u32 y, u32 ys, u32 yp, u32 mode) {
    const site_t *S = &SITES[s];
    MAC->faddr = S->off; MAC->x = x; MAC->xi = xi; MAC->xo = 0;
    MAC->y = y; MAC->ys = ys; MAC->yp = yp; MAC->nin = S->nin; MAC->nout = S->nout;
    MAC_GO((S->wbits == 4 ? MAC_W_FLASH4 : MAC_W_FLASH8) | mode);
}
// a linear layer over all frames of an FT tensor: one pass over the weights per B frames
static void lin_ft(int s, u32 x, u32 y, u32 tp, u32 yp, u32 mode) {
    for (u32 b = 0; b < tp; b += B) mac_site(s, x + b, tp, y + b, tp, yp, mode);
}

// attention scratch: raw scores (3 planes of [8][SCP]), e (int32 [8][nk]), reciprocals (u32 [8]), mixed output [8][HP]
static u32 SC, SCP, EB, RCB, MIX;
static inline i64 raw48(const i16 *p, u32 pl) {
    return ((i64)p[2 * pl] << 32) | ((i64)(u16)p[pl] << 16) | (u16)p[0];
}
// One query row against nk keys.
//   q:  288 int16 in RAM, element (h, d) at byte wq + h*wo + d*wi
//   K:  FT tensor [288][kstr];  V: VH tensor, head stride vhs elements
//   element out + (h*36 + d) * ostr = attention output, v's format.  sh = fq + fk + 4.
static void attn_row(u32 wq, u32 wi, u32 wo, u32 K, u32 kstr, u32 VH, u32 vhs, u32 nk, int sh, u32 out, u32 ostr) {
    MAC->w = wq; MAC->wi = wi; MAC->wo = wo; MAC->xi = kstr; MAC->xo = HD * kstr;
    MAC->nin = HD; MAC->nout = NH; MAC->ys = SCP; MAC->yp = NH * SCP;
    for (u32 kb = 0; kb < nk; kb += B) { MAC->x = K + kb; MAC->y = SC + kb; MAC_GO(MAC_W_RAM16 | MAC_O_RAW); }
#if USE_VEC
    VEC->src = SC; VEC->sstr = 1; VEC->srow = SCP; VEC->p1 = NH * SCP; VEC->n = nk; VEC->rows = NH;
    VEC->dst = EB; VEC->dstr = 2; VEC->drow = 2 * nk; VEC->p0 = (u32)SM_C | ((u32)sh << 16); VEC->p2 = RCB;
    VEC_GO(VEC_SOFTMAX);
#else
    i32 *E = (i32 *)P16(EB); u32 *rc = (u32 *)P16(RCB);
    for (int h = 0; h < NH; h++) {
        const i16 *p = P16(SC) + h * SCP;
        i64 mx = raw48(p, NH * SCP);
        for (u32 k = 1; k < nk; k++) { i64 s = raw48(p + k, NH * SCP); if (s > mx) mx = s; }
        u32 S = 0;
        for (u32 k = 0; k < nk; k++) {
            i64 tt = rr64((mx - raw48(p + k, NH * SCP)) * SM_C, sh);
            u64 n = (u64)tt >> 12;
            u32 e = n > 16 ? 0 : EXP2[((u32)tt & 4095) >> 4] >> (u32)n;
            E[h * nk + k] = (i32)e; S += e;
        }
        rc[h] = (u32)udiv64((u64)1 << 44, S);
    }
#endif
    // mixing: sum_k e * v, requantised by the per-head reciprocal: sat16(rshift_round(num * rc, 44))
    MAC->w = 2u * EB; MAC->wi = 4; MAC->wo = 4 * nk; MAC->xi = HP; MAC->xo = vhs;
    MAC->nin = nk; MAC->nout = NH; MAC->ys = HP; MAC->yp = 0; MAC->h = 2u * RCB; MAC->hsh = 44;
    for (u32 g = 0; g < HP; g += B) { MAC->x = VH + g; MAC->y = MIX + g; MAC_GO(MAC_W_RAM32 | MAC_O_S16 | MAC_HDRRAM); }
    v_copy(MIX, 1, HP, out, ostr, HD * ostr, HD, NH);
}

// ------------------------------------------------------------------------------------------ encoder
static u32 HLO, HPL;              // residual stream planes [288][Tp]: low at HLO, high at HLO + HPL
static u32 ENC;                   // FT [288][Tp] encoder output
static u32 CK[NLY], CVH[NLY];     // cross-attention keys (FT) and values (VH)

static void stem(u32 PCM) {
    u32 mark = brk_e;
    u32 XS = al(2016 * B), GN = al(D * T1p);
    // conv1: gather B windows of 127 samples, one pass over the weights per B frames
    for (u32 b = 0; b < T1p; b += B) {
        for (u32 l = 0; l < B; l++) {
            u32 n = b + l; if (n >= T1) n = T1 - 1;
            v_copy(PCM + 64 * n, 1, 0, XS + l, B, 0, 127, 1);
        }
        mac_site(0, XS, B, GN + b, T1p, 0, MAC_O_S16);
    }
    tr_ft(TAG(K_CONV1, 0, 0), GN, D, T1p, T1, 0);
    // tanh (Q15) and GroupNorm over the whole utterance, in place
    v_lut(GN, 1, T1p, T1, D, SITES[0].fout, 15, T_TANH, 0, 1, 1);
    tr_ft(TAG(K_TANH, 0, 0), GN, D, T1p, T1, 0);
    v_norm(GN, 0, 1, T1p, 0, GN, 1, T1p, T1, D, D * T1, N_GN, 1);
    tr_ft(TAG(K_GN, 0, 0), GN, D, T1p, T1, 0);
    trace(TAG(K_PHASE, 0, 0), 1);
    // conv2: windows of 7 frames, stride 3
    u32 C2 = al(576 * T2p);
    for (u32 b = 0; b < T2p; b += B) {
        for (u32 l = 0; l < B; l++) {
            u32 t = b + l; if (t >= T2) t = T2 - 1;
            v_copy(GN + 3 * t, 1, T1p, XS + l, B, 7 * B, 7, D);
        }
        mac_site(1, XS, B, C2 + b, T2p, 0, MAC_O_S16);
    }
    tr_ft(TAG(K_CONV2, 0, 0), C2, 576, T2p, T2, 0);
    int f2 = SITES[1].fout;
    v_lut(C2, 1, T2p, T2, 576, f2, f2, T_GELU, 1, 0, 0);
    tr_ft(TAG(K_GELU2, 0, 0), C2, 576, T2p, T2, 0);
    trace(TAG(K_PHASE, 0, 0), 2);
    // conv3: windows of 3 frames, stride 2
    u32 C3 = al(D * Tp);
    for (u32 b = 0; b < Tp; b += B) {
        for (u32 l = 0; l < B; l++) {
            u32 t = b + l; if (t >= T) t = T - 1;
            v_copy(C2 + 2 * t, 1, T2p, XS + l, B, 3 * B, 3, 576);
        }
        mac_site(2, XS, B, C3 + b, Tp, 0, MAC_O_S16);
    }
    tr_ft(TAG(K_CONV3, 0, 0), C3, D, Tp, T, 1);
    int f3 = SITES[2].fout;
    v_lut(C3, 1, Tp, T, D, f3, f3, T_GELU, 1, 0, 0);
    for (u32 f = 0; f < D; f++)
        for (u32 t = 0; t < T; t++)
            wr25(HLO + f * Tp + t, HPL, sat(rr32(P16(C3)[f * Tp + t], f3 - F_ENC_RES), RES_SAT));
    tr_res(TAG(K_STEM, 0, 0), HLO, HPL, Tp, T);
    brk_e = mark;
}

static void to_vh(u32 ft, u32 vh) {       // FT [288][Tp] -> VH [8][T][HP]
    for (u32 h = 0; h < NH; h++) v_copy(ft + h * HD * Tp, 1, Tp, vh + h * T * HP, HP, 1, T, HD);
}

static void attn_scratch(u32 nk, u32 nkp) {
    SCP = nkp; SC = al(3 * NH * SCP); EB = al(2 * NH * nk); RCB = al(2 * NH); MIX = al(NH * HP);
}

static void encoder(void) {
    u32 mark = brk_e;
    u32 U = al(5 * D * Tp), VH = al(NH * T * HP);
    attn_scratch(T, Tp);
#define UU(j) (U + (j) * D * Tp)
    for (int i = 0; i < NLY; i++) {
        for (u32 t = 0; t < T; t++) norm_row(HLO + t, HPL, Tp, UU(0) + t, Tp, N_ENCL(i, 0));
        tr_ft(TAG(K_E_LN1, i, 0), UU(0), D, Tp, T, 1);
        lin_ft(S_ENC(i, 0), UU(0), UU(1), Tp, 0, MAC_O_S16);
        lin_ft(S_ENC(i, 1), UU(0), UU(2), Tp, 0, MAC_O_S16);
        lin_ft(S_ENC(i, 2), UU(0), UU(3), Tp, 0, MAC_O_S16);
        for (u32 t = 0; t < T; t++) { v_rope(UU(1) + t, Tp, t); v_rope(UU(2) + t, Tp, t); }
        tr_ft(TAG(K_E_Q, i, 0), UU(1), D, Tp, T, 1);
        tr_ft(TAG(K_E_K, i, 0), UU(2), D, Tp, T, 1);
        tr_ft(TAG(K_E_V, i, 0), UU(3), D, Tp, T, 1);
        to_vh(UU(3), VH);
        int sh = SITES[S_ENC(i, 0)].fout + SITES[S_ENC(i, 1)].fout + 4;
        for (u32 n = 0; n < T; n++)
            attn_row(2u * (UU(1) + n), 2 * Tp, 2 * HD * Tp, UU(2), Tp, VH, T * HP, T, sh, UU(0) + n, Tp);
        tr_ft(TAG(K_E_ATTN, i, 0), UU(0), D, Tp, T, 1);
        lin_ft(S_ENC(i, 3), UU(0), HLO, Tp, HPL, MAC_O_RES | MAC_RESACC);
        tr_res(TAG(K_E_RES1, i, 0), HLO, HPL, Tp, T);
        for (u32 t = 0; t < T; t++) norm_row(HLO + t, HPL, Tp, UU(0) + t, Tp, N_ENCL(i, 1));
        tr_ft(TAG(K_E_LN2, i, 0), UU(0), D, Tp, T, 1);
        lin_ft(S_ENC(i, 4), UU(0), UU(1), Tp, 0, MAC_O_S16);
        tr_ft(TAG(K_E_FC1, i, 0), UU(1), 1152, Tp, T, 1);
        int ff = SITES[S_ENC(i, 4)].fout;
        v_lut(UU(1), 1, Tp, T, 1152, ff, ff, T_GELU, 1, 0, 0);
        tr_ft(TAG(K_E_GELU, i, 0), UU(1), 1152, Tp, T, 1);
        lin_ft(S_ENC(i, 5), UU(1), HLO, Tp, HPL, MAC_O_RES | MAC_RESACC);
        tr_res(TAG(K_E_RES2, i, 0), HLO, HPL, Tp, T);
        trace(TAG(K_PHASE, i, 0), 3);
    }
    brk_e = mark;                                   // layer scratch is dead
    ENC = al(D * Tp);
    u32 VF = al(D * Tp);
    for (u32 t = 0; t < T; t++) norm_row(HLO + t, HPL, Tp, ENC + t, Tp, N_ENC);
    tr_ft(TAG(K_ENC, 0, 0), ENC, D, Tp, T, 1);
    top_e = end_e;                                  // the residual stream is dead: cross K/V take its place
    // cross-attention keys and values, once per utterance
    for (int i = 0; i < NLY; i++) {
        CK[i] = alt(D * Tp); CVH[i] = alt(NH * T * HP);
        lin_ft(S_DEC(i, 5), ENC, CK[i], Tp, 0, MAC_O_S16);
        lin_ft(S_DEC(i, 6), ENC, VF, Tp, 0, MAC_O_S16);
        to_vh(VF, CVH[i]);
        tr_ft(TAG(K_CK, i, 0), CK[i], D, Tp, T, 1);
        tr_ft(TAG(K_CV, i, 0), VF, D, Tp, T, 1);
    }
    trace(TAG(K_PHASE, 0, 0), 4);
    brk_e = mark;                                   // the encoder output is dead too
#undef UU
}

// ------------------------------------------------------------------------------------------ decoder
static u32 KC[NLY], VC[NLY];      // self-attention cache: keys FT [288][MTp], values VH [8][MT][HP]
static int started;               // a character has been produced (the leading space is dropped)

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

// decoder linear layers: fan-in mode, one row at a time across all lanes
static void mac_vec(int s, u32 x, u32 y, u32 yp, u32 mode) { mac_site(s, x, 1, y, 1, yp, mode | MAC_FANIN); }
static void tr_hd(u32 tag, u32 hd) {
    if (!tron) return;
    u32 h = 0; for (u32 f = 0; f < D; f++) h = h * HMUL + (u32)rd25(hd + f, D);
    trace(tag, h);
}

static void decoder(void) {
    for (int i = 0; i < NLY; i++) { KC[i] = al(D * MTp); VC[i] = al(NH * MT * HP); }
    attn_scratch(T > MT ? T : MT, Tp > MTp ? Tp : MTp);
    u32 HDV = al(2 * D), YV = al(D), QV = al(D), KV = al(D), VV = al(D), AV = al(D), F1 = al(2304), PV = al(1152);
    u32 tok = BOS;
    const site_t *E = &SITES[S_EMB];
    u32 erow = 8 + (E->nin * E->wbits + 7) / 8;
    for (u32 t = 0; t < MT; t++) {
        int trs = tron && t < TRACE_STEPS;
        int tr0 = tron; tron = trs;
        // embedding row of the tied table
        fl_seek(OFF_EMBM + 4 * tok);
        i32 em = (i32)fl_u16(); int esh = (int)fl_u8();
        fl_seek(E->off + tok * erow + 8);
        if (E->wbits == 8) {
            for (u32 f = 0; f < D; f++) wr25(HDV + f, D, sat(rr64((i64)(i8)fl_u8() * em, esh), RES_SAT));
        } else {
            for (u32 f = 0; f < D; f += 2) {
                u32 by = fl_u8();
                wr25(HDV + f, D, sat(rr64((i64)((i32)(by << 28) >> 28) * em, esh), RES_SAT));
                wr25(HDV + f + 1, D, sat(rr64((i64)((i32)(by << 24) >> 28) * em, esh), RES_SAT));
            }
        }
        tr_hd(TAG(K_D_EMB, 0, t), HDV);
        for (int i = 0; i < NLY; i++) {
            norm_row(HDV, D, 1, YV, 1, N_DECL(i, 0));
            tr_v16(TAG(K_D_LN1, i, t), P16(YV), D);
            mac_vec(S_DEC(i, 0), YV, QV, 0, MAC_O_S16);
            mac_vec(S_DEC(i, 1), YV, KV, 0, MAC_O_S16);
            mac_vec(S_DEC(i, 2), YV, VV, 0, MAC_O_S16);
            v_rope(QV, 1, t); v_rope(KV, 1, t);
            tr_v16(TAG(K_D_Q, i, t), P16(QV), D);
            tr_v16(TAG(K_D_K, i, t), P16(KV), D);
            tr_v16(TAG(K_D_V, i, t), P16(VV), D);
            v_copy(KV, 1, 0, KC[i] + t, MTp, 0, D, 1);
            v_copy(VV, 1, HD, VC[i] + t * HP, 1, MT * HP, HD, NH);
            attn_row(2u * QV, 2, 2 * HD, KC[i], MTp, VC[i], MT * HP, t + 1,
                     SITES[S_DEC(i, 0)].fout + SITES[S_DEC(i, 1)].fout + 4, AV, 1);
            tr_v16(TAG(K_D_ATTN, i, t), P16(AV), D);
            mac_vec(S_DEC(i, 3), AV, HDV, D, MAC_O_RES | MAC_RESACC);
            tr_hd(TAG(K_D_RES1, i, t), HDV);
            norm_row(HDV, D, 1, YV, 1, N_DECL(i, 1));
            tr_v16(TAG(K_D_LN2, i, t), P16(YV), D);
            mac_vec(S_DEC(i, 4), YV, QV, 0, MAC_O_S16);
            tr_v16(TAG(K_D_CQ, i, t), P16(QV), D);
            attn_row(2u * QV, 2, 2 * HD, CK[i], Tp, CVH[i], T * HP, T,
                     SITES[S_DEC(i, 4)].fout + SITES[S_DEC(i, 5)].fout + 4, AV, 1);
            tr_v16(TAG(K_D_CATTN, i, t), P16(AV), D);
            mac_vec(S_DEC(i, 7), AV, HDV, D, MAC_O_RES | MAC_RESACC);
            tr_hd(TAG(K_D_RES2, i, t), HDV);
            norm_row(HDV, D, 1, YV, 1, N_DECL(i, 2));
            tr_v16(TAG(K_D_LN3, i, t), P16(YV), D);
            mac_vec(S_DEC(i, 8), YV, F1, 0, MAC_O_S16);
            tr_v16(TAG(K_D_FC1, i, t), P16(F1), 2304);
            int ff = SITES[S_DEC(i, 8)].fout;
            i16 *f1 = P16(F1), *pv = P16(PV);
            v_lut(F1 + 1152, 1, 0, 1152, 1, ff, ff, T_SILU, 1, 0, 0);
            tr_v16(TAG(K_D_SILU, i, t), f1 + 1152, 1152);
            v_mul(F1, 1152, PV, 1152, 2 * ff - F_PROD[i]);
            tr_v16(TAG(K_D_PROD, i, t), pv, 1152);
            mac_vec(S_DEC(i, 9), PV, HDV, D, MAC_O_RES | MAC_RESACC);
            tr_hd(TAG(K_D_RES3, i, t), HDV);
        }
        norm_row(HDV, D, 1, YV, 1, N_DEC);
        tr_v16(TAG(K_D_LN, 0, t), P16(YV), D);
        mac_vec(S_EMB, YV, 0, 0, MAC_O_ARGMAX);
        tok = MAC->argidx;
        if (trs) { trace(TAG(K_D_ARGIDX, 0, t), tok); trace(TAG(K_D_ARGVAL, 0, t), MAC->argval); }
        tron = tr0;
        trace(TAG(K_TOKEN, 0, t), tok);
        if (tok == EOS) break;
        detok(tok);
    }
}

// ------------------------------------------------------------------------------------------ top
#ifndef MAX_SEC
#define MAX_SEC 30           // window cap: the longest stretch of audio transcribed in one piece; bounds the RAM
#endif
int model_main(void) {
    tron = hal_trace_on();
    brk_e = (hal_heap_start() + 2 * B - 1) / (2 * B) * B; hiw_e = brk_e;
    end_e = hal_heap_end() / (2 * B) * B; top_e = end_e;
    u32 base = brk_e, PCM = brk_e, st, cap = end_e - PCM;
    if (cap > 16000u * MAX_SEC) cap = 16000u * MAX_SEC;
    N = hal_capture(P16(PCM), cap, &st);
    if (N == 0 && st == ~0u) return 0;
    started = 0;
    trace(TAG(K_START, 0, 0), st);
    trace(TAG(K_CAP, 0, 0), hal_cap_info());
    al(N);
    trace(TAG(K_NSAMP, 0, 0), N);
    tr_v16(TAG(K_PCM, 0, 0), P16(PCM), N);
    if (N < 127 + 64 * 12) { uart_putc('\n'); trace(TAG(K_DONE, 0, 0), 0); return 1; }
    T1 = (N - 127) / 64 + 1; T2 = (T1 - 7) / 3 + 1; T = (T2 - 3) / 2 + 1;
    T1p = UP(T1); T2p = UP(T2); Tp = UP(T);
    MT = (N * 13 + 31999) / 32000; if (MT < 1) MT = 1;
    MTp = UP(MT);
    HPL = D * Tp; HLO = alt(2 * D * Tp);           // residual stream: top, through the stem and the encoder
    stem(PCM);
    brk_e = base;                                   // the audio is dead after conv1
    encoder();
    decoder();
    uart_putc('\n');
    trace(TAG(K_MEM, 0, 0), 2 * hiw_e);
    trace(TAG(K_SAT, 0, 0), MAC->satflag);
    trace(TAG(K_DONE, 0, 0), 1);
    return 1;
}
