// PC build of the controller firmware. The MAC array, the per-element unit, the weight source and the pins
// are C functions here; model.c is the same file the chip runs. Used to check the program against the golden
// model and as the functional reference for imh_mac.sv (mac_go) and imh_vec.sv (vec_go).
//
//   fw_host <data.bin> <pcm.raw int16> <events.txt> [trace=1] [ram_bytes]
//
// events.txt: one "tag value" pair per line (hex), the same stream the chip puts on its trace port.
// stdout: the UART text. stderr: counters.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "hal.h"
#include "layout.h"
#define DATA_BASE_H 0x10000u

u8 *ram;
static mac_regs_t regs;
static vec_regs_t vregs;
volatile mac_regs_t *MAC = &regs;
volatile vec_regs_t *VEC = &vregs;
static fan_regs_t fregs;
volatile fan_regs_t *FAN = &fregs;
static u8 *flash; static u32 flash_len, fl_pos;
static i16 *pcm_in; static u32 pcm_n;
static FILE *ev;
static int trace_on = 1;
static u32 ram_bytes = 64u << 20;
static u64 n_ops, n_mac_flash, n_mac_ram, n_fl_bytes, n_seeks, n_rows, n_vops, n_vel;
static i64 v_s; static u64 v_q;
static u32 n_rec_hi;

static i64 rr64(i64 x, int s) { return s <= 0 ? x << (-s) : (x + ((i64)1 << (s - 1))) >> s; }
static inline i16 *el(u32 e) { if (2u * e + 1 >= ram_bytes) { fprintf(stderr, "RAM element %u out of range\n", e); exit(2); } return (i16 *)(ram + 2u * e); }

// ---------------------------------------------------------------------------------------- MAC array
void mac_go(u32 ctrl) {
    u32 wsrc = ctrl & 3, om = (ctrl >> 2) & 3, scalar = (ctrl >> 4) & 1, fanin = (ctrl >> 5) & 1;
    u32 resacc = (ctrl >> 6) & 1, hdrram = (ctrl >> 7) & 1, single = scalar | fanin, x8 = (ctrl >> 8) & 1;
    u32 fp = regs.faddr; int nib = 0;
    i64 best = 0; u32 besti = 0;
    n_ops++;
    if (wsrc < 2) n_seeks++;
    if (fanin && (regs.x % LANES || regs.xo % LANES)) { fprintf(stderr, "fan-in x not word aligned\n"); exit(2); }
    for (u32 o = 0; o < regs.nout; o++) {
        i64 acc[LANES]; i64 M = 1; int sh = 0;
        i64 bias = 0;
        if (wsrc < 2) {
            const u8 *h = flash + fp; fp += 8; n_fl_bytes += 8; nib = 0;
            u64 b = 0; for (int j = 0; j < 5; j++) b |= (u64)h[j] << (8 * j);
            bias = (i64)(b << 24) >> 24; M = h[5] | (h[6] << 8); sh = h[7];
        } else if (hdrram) { M = *(u32 *)(ram + regs.h + 4 * o); sh = (int)regs.hsh; }
        for (int l = 0; l < LANES; l++) acc[l] = bias;
        for (u32 i = 0; i < regs.nin; i++) {
            u32 xe = regs.x + o * regs.xo + (regs.xw && !fanin ? (regs.xs0 + i * regs.xi) % regs.xw : i * (fanin ? 1 : regs.xi));
            i64 w;
            if (wsrc == MAC_W_FLASH8) { w = (i8)flash[fp++]; n_fl_bytes++; }
            else if (wsrc == MAC_W_FLASH4) {
                u8 by = flash[fp]; w = nib ? (i8)by >> 4 : (i8)(by << 4) >> 4;
                if (nib) { fp++; n_fl_bytes++; } nib ^= 1;
            } else {
                u32 wa = regs.w + o * regs.wo + i * regs.wi;
                if (wsrc == MAC_W_RAM16) w = *(i16 *)(ram + wa);
                else { i32 v = *(i32 *)(ram + wa); w = (v << 14) >> 14; if (w != v) { fprintf(stderr, "ram32 weight %d over 18 bits\n", v); exit(2); } }
            }
            if (x8) {
                if (xe >= ram_bytes) { fprintf(stderr, "RAM byte %u out of range\n", xe); exit(2); }
                if (single) acc[0] += (i64)(i8)ram[xe] * w;
                else for (int l = 0; l < LANES; l++) acc[l] += (i64)(i8)ram[(xe & ~(u32)(LANES - 1)) + l] * w;
            }
            else if (single) acc[0] += (i64)*el(xe) * w;
            else { const i16 *xp = el(xe & ~(u32)(LANES - 1)); el((xe & ~(u32)(LANES - 1)) + LANES - 1);
                   for (int l = 0; l < LANES; l++) acc[l] += (i64)xp[l] * w; }
            if (wsrc < 2) n_mac_flash++; else n_mac_ram++;
        }
        if (wsrc == MAC_W_FLASH4 && nib) { fp++; n_fl_bytes++; }
        n_rows++;
        u32 ye = regs.y + o * regs.ys;
        int l1 = (single || om == 3) ? 1 : LANES;                 // argmax looks at lane 0 only
        if (!single) ye &= ~(u32)(LANES - 1);
        for (int l = 0; l < l1; l++) {
            i64 a = acc[l];
            if (a >= ((i64)1 << 47) || a < -((i64)1 << 47)) { fprintf(stderr, "accumulator over 48 bits\n"); exit(2); }
            if (om == 0) { i64 v = rr64(a * M, sh); v = v > 32767 ? 32767 : v < -32767 ? -32767 : v; *el(ye + l) = (i16)v; }
            else if (om == 1) {
                i64 v = rr64(a * M, sh); i64 m = (1 << 24) - 1; v = v > m ? m : v < -m ? -m : v;
                if (resacc) {
                    v += (i32)(((u32)(u16)*el(ye + regs.yp + l) << 16) | (u16)*el(ye + l));
                    v = v > m ? m : v < -m ? -m : v;
                }
                *el(ye + l) = (i16)(v & 0xFFFF); *el(ye + regs.yp + l) = (i16)(v >> 16);
            }
            else if (om == 2) { *el(ye + l) = (i16)(a & 0xFFFF); *el(ye + regs.yp + l) = (i16)((a >> 16) & 0xFFFF);
                                *el(ye + 2 * regs.yp + l) = (i16)((a >> 32) & 0xFFFF); }
            else { i64 v = rr64(a * M, sh); if (o == 0 || v > best) { best = v; besti = o; } }
        }
    }
    if (om == 3) { regs.argidx = besti; regs.argval = (u32)best; }
}

// ---------------------------------------------------------------------------------------- fan-out array
// Reference of rtl/core/imh_fan.sv. Weight stream per group of FAN_L rows: FAN_L headers of 8 bytes, then per input
// FAN_L weights (int8 bytes, or int4 two per byte, low nibble first).
void fan_go(u32 ctrl) {
    const fan_regs_t R = fregs;
    u32 w4 = ctrl & 1, w16 = (ctrl >> 1) & 1, om = (ctrl >> 2) & 3, resacc = (ctrl >> 6) & 1;
    u32 fp = R.faddr, besti[FAN_CMAX] = {0};
    i64 best[FAN_CMAX] = {0};
    static i64 acc[FAN_CMAX][FAN_L];
    if (R.nc < 1 || R.nc > FAN_CMAX || om == 2 || R.x % 16 || R.xs % 16 || R.y % 16 ||
        R.ys % 16 || R.yp % 16 || R.nin % 16 || R.nin == 0) { fprintf(stderr, "fan: bad parameters\n"); exit(2); }
    n_ops++; n_seeks++;
    for (u32 g = 0; g < R.nout; g += FAN_L) {
        i64 bias[FAN_L], M[FAN_L]; int sh[FAN_L];
        for (int l = 0; l < FAN_L; l++) {
            const u8 *h = flash + fp + 8 * l; u64 b = 0;
            for (int j = 0; j < 5; j++) b |= (u64)h[j] << (8 * j);
            bias[l] = (i64)(b << 24) >> 24; M[l] = h[5] | (h[6] << 8); sh[l] = h[7] & 63;
        }
        fp += 8 * FAN_L; n_fl_bytes += 8 * FAN_L;
        for (u32 c = 0; c < R.nc; c++) for (int l = 0; l < FAN_L; l++) acc[c][l] = bias[l];
        for (u32 i = 0; i < R.nin; i++) {
            i64 w[FAN_L];
            for (int l = 0; l < FAN_L; l++)
                w[l] = w4 ? (i64)((i8)(u8)(flash[fp + l / 2] << ((l & 1) ? 0 : 4)) >> 4)
                     : w16 ? (i64)(i16)(flash[fp + 2 * l] | (flash[fp + 2 * l + 1] << 8)) : (i64)(i8)flash[fp + l];
            fp += w4 ? FAN_L / 2 : w16 ? 2 * FAN_L : FAN_L; n_fl_bytes += w4 ? FAN_L / 2 : w16 ? 2 * FAN_L : FAN_L;
            for (u32 c = 0; c < R.nc; c++) {
                i64 x = *el(R.x + c * R.xs + i);
                for (int l = 0; l < FAN_L; l++) acc[c][l] += x * w[l];
            }
            n_mac_flash += (u64)R.nc * FAN_L;
        }
        n_rows++;
        for (u32 c = 0; c < R.nc; c++) for (int l = 0; l < FAN_L; l++) {
            u32 o = g + l, ye = R.y + c * R.ys + o;
            if (o >= R.nout) continue;
            i64 a = acc[c][l];
            if (a >= ((i64)1 << 47) || a < -((i64)1 << 47)) { fprintf(stderr, "accumulator over 48 bits\n"); exit(2); }
            i64 v = rr64(a * M[l], sh[l]);
            if (om == 0) { v = v > 32767 ? 32767 : v < -32767 ? -32767 : v; *el(ye) = (i16)v; }
            else if (om == 1) {
                i64 m = (1 << 24) - 1; v = v > m ? m : v < -m ? -m : v;
                if (resacc) { v += (i32)(((u32)(u16)*el(ye + R.yp) << 16) | (u16)*el(ye)); v = v > m ? m : v < -m ? -m : v; }
                *el(ye) = (i16)(v & 0xFFFF); *el(ye + R.yp) = (i16)(v >> 16);
            }
            else if (o == 0 || v > best[c]) { best[c] = v; besti[c] = o; }
        }
    }
    if (om == 3) { fregs.argidx = besti[0]; fregs.argval = (u32)best[0]; for (u32 c = 0; c < R.nc; c++) fregs.argc[c] = besti[c]; }
}

// ---------------------------------------------------------------------------------------- per-element unit
static i64 lut_ref(i64 x, int f, const int *tab, int tails, int f_out, int r4) {
    int fmin = r4 ? 4 : 5, fw = f > fmin ? f : fmin;
    x <<= (fw - f);
    i64 lim = (i64)(r4 ? 16 : 8) << fw;
    i64 u = x < -lim ? -lim : x > lim - 1 ? lim - 1 : x;
    u += lim;
    int sh = fw - fmin;
    i64 idx = u >> sh, fr = u & (((i64)1 << sh) - 1);
    i64 y = rr64(tab[idx] + (((tab[idx + 1] - tab[idx]) * fr) >> sh), 14 - f_out);
    if (tails) {
        if (x >= lim) y = rr64(x, fw - f_out);
        if (x < -lim) y = 0;
    }
    return y;
}
static u64 isqrt_ref(u64 n) {
    u64 res = 0, bit = (u64)1 << 60;
    for (int i = 0; i < 31; i++) { u64 t = res + bit; if (n >= t) { n -= t; res = (res >> 1) + bit; } else res >>= 1; bit >>= 2; }
    return res;
}
void vec_go(u32 ctrl) {
    const vec_regs_t R = vregs;
    u32 op = ctrl & 7, wide = (ctrl >> 4) & 1, gbrow = (ctrl >> 5) & 1, autop = (ctrl >> 6) & 1;
    if (ctrl & VEC_CLR) { v_s = 0; v_q = 0; }
    n_vops++;
    if (op == VEC_NSTAT) {
        i64 n = R.p0, num = 2 * v_s + n;
        i64 mean = num >= 0 ? (i64)((u64)num / (u64)(2 * n)) : -(i64)(((u64)(-num) + (u64)(2 * n) - 1) / (u64)(2 * n));
        mean = (i32)mean;
        u64 ss = v_q - (u64)(2 * mean * v_s) + (u64)n * (u64)(mean * mean);
        u64 var = ss / (u64)n + R.p1; if (var < 1) var = 1;
        int bl = 0; for (u64 v = var; v; v >>= 1) bl++;
        int s = ((61 - bl) >> 1) * 2;
        u64 sd = isqrt_ref(s >= 0 ? var << s : var >> (-s));
        vregs.mean = (u32)mean; vregs.rec = (u32)((((u64)1 << 61) / sd) & 0xFFFFFFFFu); vregs.sh1 = (u32)(49 - s / 2) & 63;
        n_rec_hi = (u32)((((u64)1 << 61) / sd) >> 32) & 1;
        return;
    }
    if (op == VEC_ROPE) {
        const u8 *t = flash + R.faddr; n_fl_bytes += 64;
        for (u32 r = 0; r < R.rows; r++) for (u32 i = 0; i < R.n; i++) {
            u32 e = R.src + r * R.srow + i * R.sstr;
            i64 c = (i16)(t[2 * (i & 15)] | (t[2 * (i & 15) + 1] << 8)), s = (i16)(t[32 + 2 * (i & 15)] | (t[33 + 2 * (i & 15)] << 8));
            i64 x1 = *el(e), x2 = *el(e + R.p1);
            i64 y1 = rr64(x1 * c - x2 * s, 14), y2 = rr64(x2 * c + x1 * s, 14);
            *el(e) = (i16)(y1 > 32767 ? 32767 : y1 < -32767 ? -32767 : y1);
            *el(e + R.p1) = (i16)(y2 > 32767 ? 32767 : y2 < -32767 ? -32767 : y2);
            n_vel += 2;
        }
        return;
    }
    u32 fp = R.faddr; i64 G = 0, Bv = 0;
    for (u32 r = 0; r < R.rows; r++) {
        u32 sb = R.src + r * R.srow, db = R.dst + r * R.drow;
        if (op == VEC_SOFTMAX) {
            u32 pl = R.p1; int sh = (R.p0 >> 16) & 63; i64 C = R.p0 & 0xFFFF;
#define RAW(k) (((i64)*el(sb + (k) * R.sstr + 2 * pl) << 32) | ((i64)(u16)*el(sb + (k) * R.sstr + pl) << 16) | (u16)*el(sb + (k) * R.sstr))
            i64 mx = RAW(0);
            for (u32 k = 1; k < R.n; k++) { i64 s = RAW(k); if (s > mx) mx = s; }
            u64 S = 0;
            for (u32 k = 0; k < R.n; k++) {
                u64 tt = (u64)rr64((mx - RAW(k)) * C, sh);
                u64 nn = tt >> 12;
                u32 e = nn > 16 ? 0 : EXP2[(tt & 4095) >> 4] >> nn;
                *(i32 *)el(db + k * R.dstr) = (i32)e; el(db + k * R.dstr + 1);
                S += e;
            }
            *(u32 *)el(R.p2 + 2 * r) = (u32)(((u64)1 << 44) / S);
            n_vel += 2 * R.n;
            continue;
        }
        for (u32 i = 0; i < R.n; i++) {
            u32 se = sb + i * R.sstr, de = db + i * R.dstr;
            n_vel++;
            if (op == VEC_COPY && (ctrl & VEC_B8)) {
                i64 v = rr64(*el(se), (int)(R.p0 & 31)); v = v > 127 ? 127 : v < -127 ? -127 : v;
                if (de >= ram_bytes) { fprintf(stderr, "RAM byte %u out of range\n", de); exit(2); }
                ram[de] = (u8)(i8)v;
            }
            else if (op == VEC_COPY) *el(de) = *el(se);
            else if (op == VEC_MUL) { i64 v = rr64((i64)*el(se) * *el(se + R.p1), (int)(R.p0 & 31)); *el(de) = (i16)(v > 32767 ? 32767 : v < -32767 ? -32767 : v); }
            else if (op == VEC_LUT) {
                int f = R.p0 & 15, fo = (R.p0 >> 4) & 15, tab = (R.p0 >> 8) & 3;
#ifdef LUT_ASINH_N
                const int *T = tab == 0 ? LUT_TANH : tab == 1 ? LUT_GELU : tab == 2 ? LUT_SILU : LUT_ASINH;
#else
                const int *T = tab == 0 ? LUT_TANH : tab == 1 ? LUT_GELU : LUT_SILU;
#endif
                i64 y = lut_ref(*el(se), f, T, (R.p0 >> 10) & 1, fo, (R.p0 >> 13) & 1);
                if ((R.p0 >> 11) & 1) y = y > 32767 ? 32767 : y < -32767 ? -32767 : y;
                *el(de) = (i16)y;
                if ((R.p0 >> 12) & 1) { v_s += y; v_q += (u64)(y * y); }
            } else {
                i64 x = wide ? (i32)(((u32)(u16)*el(se + R.p1) << 16) | (u16)*el(se)) : *el(se);
                if (op == VEC_STAT) { v_s += x; v_q += (u64)(x * x); }
                else if (op == VEC_NORM) {
                    if (!gbrow || i == 0) {
                        G = (i16)(flash[fp] | (flash[fp + 1] << 8));
                        Bv = (i32)((u32)flash[fp + 2] | ((u32)flash[fp + 3] << 8) | ((u32)flash[fp + 4] << 16) | ((u32)flash[fp + 5] << 24));
                        fp += 6; n_fl_bytes += 6;
                    }
                    int sh1 = autop ? (int)vregs.sh1 : (int)(R.p3 & 63), sh2 = (R.p3 >> 8) & 63;
                    i64 mean = autop ? (i32)vregs.mean : (i32)R.p0;
                    u64 rec = autop ? (((u64)n_rec_hi << 32) | vregs.rec) : (u64)R.p2;
                    i32 y = (i32)rr64((x - mean) * (i64)rec, sh1);
                    i64 o = rr64((i64)y * G, sh2) + Bv;
                    *el(de) = (i16)(o > 32767 ? 32767 : o < -32767 ? -32767 : o);
                }
            }
        }
    }
    vregs.s_lo = (u32)v_s; vregs.s_hi = (u32)((u64)v_s >> 32); vregs.q_lo = (u32)v_q; vregs.q_hi = (u32)(v_q >> 32);
}

void fl_seek(u32 a) { fl_pos = a; n_seeks++; }
u32 fl_u8(void) { if (fl_pos >= flash_len) { fprintf(stderr, "flash read past end\n"); exit(2); } n_fl_bytes++; return flash[fl_pos++]; }
void uart_putc(int c) { putchar(c); }
void trace(u32 tag, u32 val) { if (ev) fprintf(ev, "%08x %08x\n", tag, val); }
int hal_trace_on(void) { return trace_on; }
u32 hal_heap_start(void) { return 0; }
u32 hal_heap_end(void) { return ram_bytes; }
static int captured;
u32 hal_cap_info(void) { return 0; }
static u32 seg_pos, seg_open;
i32 hal_seg_begin(void) { if (captured++) return -1; seg_pos = 0; seg_open = 1; return 0; }
int hal_sample(i16 *v) { if (!seg_open || seg_pos >= pcm_n) { seg_open = 0; return 0; } *v = pcm_in[seg_pos++]; return 1; }
u32 hal_backlog(void) { return 0; }  // no timing on the PC
u32 hal_capture(i16 *dst, u32 max, u32 *start) {
    if (captured++) { *start = ~0u; return 0; }
    u32 n = pcm_n < max ? pcm_n : max; memcpy(dst, pcm_in, 2 * n); *start = 0; return n;
}
void hal_fail(u32 code) { fprintf(stderr, "hal_fail %u\n", code); exit(3); }

// library use (unit tests): the C references on caller-provided memories
void host_lib_init(u8 *r, u32 rbytes, u8 *f, u32 flen) { ram = r; ram_bytes = rbytes; flash = f; flash_len = flen; }

#ifndef HOST_LIB
static u8 *slurp(const char *p, u32 *n) {
    FILE *f = fopen(p, "rb"); if (!f) { perror(p); exit(1); }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    u8 *b = malloc(sz + 16); if (fread(b, 1, sz, f) != (size_t)sz) exit(1); fclose(f); *n = (u32)sz; return b;
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: fw_host data.bin pcm.raw events.txt [trace] [ram_bytes]\n"); return 1; }
    u32 dn; u8 *d = slurp(argv[1], &dn);
    flash_len = DATA_BASE_H + dn; flash = malloc(flash_len); memset(flash, 0xFF, DATA_BASE_H); memcpy(flash + DATA_BASE_H, d, dn);
    u32 pn; pcm_in = (i16 *)slurp(argv[2], &pn); pcm_n = pn / 2;
    ev = fopen(argv[3], "w");
    if (argc > 4) trace_on = atoi(argv[4]);
    if (argc > 5) ram_bytes = (u32)strtoul(argv[5], 0, 0);
    ram = malloc(ram_bytes); memset(ram, 0xA5, ram_bytes);
    while (model_main()) ;
    fclose(ev);
    fprintf(stderr, "mac_ops %llu rows %llu mac_stream %llu mac_ram %llu weight_bytes %llu seeks %llu vec_ops %llu vec_elements %llu\n",
            n_ops, n_rows, n_mac_flash, n_mac_ram, n_fl_bytes, n_seeks, n_vops, n_vel);
    return 0;
}
#endif
