// Unit test of imh_mac (MAC array, row sequencer, requantiser, argmax) and imh_vec (per-element unit) against
// their C references mac_go() and vec_go() in fw/hal_host.c.
// Random operations over every weight source (int8 stream, int4 stream, RAM 16 bit, RAM 32 bit), every output
// mode (sat16, sat25 with and without accumulate, raw, argmax), batch / scalar / fan-in addressing, RAM
// multipliers, int8 x operands (batch and scalar), and every per-element operation (copy, copy to int8, LUT,
// statistics, norm, softmax, rope, multiply), with a weight source that offers a random number of bytes (0..LANES)
// each clock. After each operation the whole RAM and the result registers are compared. Built against the
// streaming image (layout.h with LUT_ASINH_N, Makefile test-mac), the LUT operations use the GELU, SiLU and asinh
// tables (the latter with the range 2^4 flag) instead of Imhotep's tanh, GELU and SiLU.
//
//   test_units [n_ops] [seed]        exit code 0 = all operations identical
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "Vunit_tb.h"
#include "verilated.h"
#include "hal.h"            // hal_host.c is compiled as C++ in this build
#include "layout.h"         // LUT_ASINH_N in the streaming image
void host_lib_init(u8 *r, u32 rbytes, u8 *f, u32 flen);
#define RAMB 16384
#define FLB 65536
#define WBY (2 * LANES)

static Vunit_tb *top;
static unsigned long long cyc;
static u8 mram[RAMB], fl[FLB + 64];
static u32 flptr, flav;
static u32 rs = 1;
static u32 rnd() { rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5; return rs; }
static int stall_pct, wide_src;

static void tick() {
    top->eval();                     // inputs set since the last tick settle with clk low (clock-gate latches see them)
    u32 take = top->ws_take; bool seek = top->seek_valid; u32 sa = top->seek_addr;
    top->clk = 1; top->eval();
    if (seek) { flptr = sa; flav = 0; }
    else if (take) { flptr += take; flav = 0; }
    if (!flav && (int)(rnd() % 100) >= stall_pct) flav = wide_src ? 1 + rnd() % LANES : 1;
    top->clk = 0; top->eval();
    top->ws_avail = flav; memcpy(&top->ws_data, fl + (flptr % FLB), LANES);
    top->eval();
    cyc++;
}
static void wr_reg(int unit, int idx, u32 v) {
    top->reg_we_m = unit == 0; top->reg_we_v = unit == 1; top->reg_idx = idx; top->reg_wdata = v; tick();
    top->reg_we_m = 0; top->reg_we_v = 0;
}
static void ram_load() {
    for (u32 w = 0; w < RAMB / WBY; w++) { top->h_en = 1; top->h_we = 1; top->h_addr = w; memcpy(&top->h_wdata, mram + WBY * w, WBY); tick(); }
    top->h_en = 0; top->h_we = 0;
}
static int ram_compare() {
    int bad = 0;
    top->h_en = 1; top->h_we = 0; top->h_addr = 0; tick();
    for (u32 w = 0; w < RAMB / WBY; w++) {
        top->h_addr = w + 1;
        if (memcmp(&top->h_rdata, mram + WBY * w, WBY)) {
            if (!bad) {
                fprintf(stderr, "  RAM word %u (element %u..) differs\n", w, w * LANES);
                const short *g = (const short *)&top->h_rdata, *e = (const short *)(mram + WBY * w);
                for (int k = 0; k < LANES; k++) if (g[k] != e[k]) fprintf(stderr, "    element %u: rtl %d ref %d\n", w * LANES + k, g[k], e[k]);
            }
            bad++;
        }
        tick();
    }
    top->h_en = 0;
    return bad;
}
static void wait_done(bool mac) {
    unsigned long long c0 = cyc;
    while (!(mac ? top->done_m : top->done_v)) { tick(); if (cyc - c0 > 4000000) { fprintf(stderr, "timeout\n"); exit(2); } }
    tick();
}
static void put16(u32 e, int v) { i16 x = (i16)v; memcpy(mram + 2 * e, &x, 2); }
static void put32(u32 byte, i32 v) { memcpy(mram + byte, &v, 4); }

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    int nops = argc > 1 ? atoi(argv[1]) : 3000;
    if (argc > 2) rs = (u32)atoi(argv[2]) | 1;
    top = new Vunit_tb;
    host_lib_init(mram, RAMB, fl, FLB);
    for (u32 i = 0; i < RAMB; i += 2) put16(i / 2, (rnd() % 10 == 0) ? ((rnd() & 1) ? 32767 : -32768) : (int)rnd());
    top->clk = 0; top->rst_n = 0; top->reg_we_m = 0; top->reg_we_v = 0; top->go_m = 0; top->go_v = 0; top->h_en = 0; top->h_we = 0; top->ws_avail = 0;
    for (int i = 0; i < 4; i++) tick();
    top->rst_n = 1; tick();
    ram_load();
    int fails = 0, nx8 = 0, nb8 = 0, nasinh = 0; long long macs = 0, vels = 0; int cnt[4][4] = {{0}}, fcnt = 0, vcnt[7] = {0};
    const u32 NE = RAMB / 2;
    for (int op = 0; op < nops && fails < 5; op++) {
        stall_pct = (rnd() % 3 == 0) ? 0 : (int)(rnd() % 80);
        wide_src = rnd() % 4 != 0;
        int bad = 0; char what[160];
        bool reload = false;
        if (rnd() % 3) {
            // ------------------------------------------------ MAC array
            u32 wsrc = rnd() % 4, om = rnd() % 4, scalar = rnd() % 2, fanin = 0, resacc = 0, hdrram = 0;
            if (wsrc < 2 && rnd() % 2) { fanin = 1; scalar = 0; }
            if (wsrc >= 2 && rnd() % 3 == 0) hdrram = 1;
            u32 x8 = (!fanin && rnd() % 4 == 0) ? 1 : 0;           // int8 x, addressed in bytes
            if (wsrc >= 2 && !hdrram && om != 2 && rnd() % 4) om = 2;
            if (om == 1 && rnd() % 2) resacc = 1;
            u32 nin = 1 + rnd() % 70, nout = 1 + rnd() % 6;
            bool single = scalar || fanin;
            mac_regs_t r; memset(&r, 0, sizeof r);
            r.nin = nin; r.nout = nout;
            if (fanin) { r.xi = 1; r.xo = (rnd() % 3) * LANES; r.x = (rnd() % (NE / 2 - nin - nout * r.xo - LANES)) & ~(u32)(LANES - 1); }
            else {
                r.xi = scalar ? rnd() % 5 : (rnd() % 4) * LANES; r.xo = scalar ? rnd() % 7 : (rnd() % 3) * LANES;
                r.x = rnd() % (NE / 2 - nin * r.xi - nout * r.xo - LANES);
                if (rnd() % 3 == 0) {                           // ring: x offset (xs0 + i*xi) mod xw
                    r.xw = scalar ? r.xi + 1 + rnd() % 40 : LANES * (r.xi / LANES + 1 + rnd() % 4);
                    r.xs0 = scalar ? rnd() % r.xw : LANES * (rnd() % (r.xw / LANES));
                    r.x = rnd() % (NE / 2 - r.xw - nout * r.xo - LANES);
                }
            }
            r.wi = (wsrc == 3 ? 4 : 2) * (rnd() % 4); r.wo = (wsrc == 3 ? 4 : 2) * (rnd() % 5);
            r.w = (rnd() % (RAMB / 2 - nin * r.wi - nout * r.wo - 8)) & ~(wsrc == 3 ? 3u : 1u);
            r.ys = single ? 1 + rnd() % 3 : LANES * (1 + rnd() % 2);
            r.yp = single ? nout * r.ys + rnd() % 3 : LANES * (nout * (r.ys / LANES) + rnd() % 2);
            r.y = NE / 2 + rnd() % (NE / 2 - 3 * r.yp - nout * r.ys - LANES);
            r.faddr = rnd() % (FLB - 600);
            r.h = RAMB - 64; r.hsh = 1 + rnd() % 44;
            if (wsrc < 2) {
                u32 p = r.faddr;
                for (u32 o = 0; o < nout; o++) {
                    long long bias = (long long)(int)rnd() >> (rnd() % 24); u32 M = 16384 + rnd() % 16384, sh = 1 + rnd() % 40;
                    for (int j = 0; j < 5; j++) fl[p++] = (u8)(bias >> (8 * j));
                    fl[p++] = M & 255; fl[p++] = M >> 8; fl[p++] = sh;
                    u32 nb = wsrc == 0 ? nin : (nin + 1) / 2;
                    for (u32 j = 0; j < nb; j++) fl[p++] = (u8)rnd();
                }
            } else {
                if (wsrc == 3) for (u32 o = 0; o < nout; o++) for (u32 i = 0; i < nin; i++)
                    put32(r.w + o * r.wo + i * r.wi, (rnd() % 8 == 0) ? 65536 : (i32)(rnd() % 262144) - 131072);
                if (hdrram) for (u32 o = 0; o < nout; o++) put32(r.h + 4 * o, (i32)(1 + rnd() % (rnd() % 2 ? 60000 : 1000000)));
                reload = wsrc == 3 || hdrram;
            }
            if (reload) ram_load();
            u32 ctrl = wsrc | (om << 2) | (scalar << 4) | (fanin << 5) | (resacc << 6) | (hdrram << 7) | (x8 << 8);
            const u32 *rv = (const u32 *)&r;
            for (int k = 0; k < 12; k++) wr_reg(0, k, rv[k]);
            wr_reg(0, 16, r.h); wr_reg(0, 17, r.hsh); wr_reg(0, 18, r.xw); wr_reg(0, 19, r.xs0);
            top->go_m = 1; top->reg_wdata = ctrl; tick(); top->go_m = 0;
            wait_done(true);
            memcpy((void *)MAC, &r, sizeof r); mac_go(ctrl);
            if (om == 3 && (top->argidx != MAC->argidx || top->argval != MAC->argval)) { fprintf(stderr, "  argmax %u/%08x vs %u/%08x\n", top->argidx, top->argval, MAC->argidx, MAC->argval); bad++; }
            if (top->satflag) { fprintf(stderr, "  saturation flag set\n"); bad++; }
            snprintf(what, sizeof what, "mac wsrc %u om %u scalar %u fanin %u resacc %u hdrram %u x8 %u nin %u nout %u", wsrc, om, scalar, fanin, resacc, hdrram, x8, nin, nout);
            nx8 += x8;
            macs += (long long)nin * nout; cnt[wsrc][om]++; fcnt += fanin;
        } else {
            // ------------------------------------------------ per-element unit
            u32 vop = rnd() % 8, flags = 0;
            vec_regs_t r; memset(&r, 0, sizeof r);
            r.n = 1 + rnd() % 30; r.rows = 1 + rnd() % 4;
            r.sstr = rnd() % 4; r.srow = rnd() % 40; r.dstr = 1 + rnd() % 3; r.drow = 40 + rnd() % 40;
            r.src = rnd() % (NE / 2 - 400); r.dst = NE / 2 + rnd() % (NE / 2 - 1000);
            r.p1 = 150 + rnd() % 20;
            if (vop == VEC_LUT) {
#ifdef LUT_ASINH_N
                u32 tab = 1 + rnd() % 3, r4 = rnd() % 2;           // GELU, SiLU, asinh; range 2^4 or 2^3
                if (tab == 3) nasinh++;
#else
                u32 tab = rnd() % 3, r4 = rnd() % 4 == 0;
#endif
                r.p0 = (rnd() % 16) | ((rnd() % 16) << 4) | (tab << 8) | ((rnd() % 2) << 10) | ((rnd() % 2) << 11) | ((rnd() % 2) << 12) | (r4 << 13);
                if (rnd() % 2) flags |= VEC_CLR;
                if (rnd() % 2) { r.dst = r.src; r.dstr = r.sstr = 1 + rnd() % 3; r.drow = r.srow; }
            } else if (vop == VEC_STAT) {
                if (rnd() % 2) flags |= VEC_WIDE;
                if (rnd() % 2) flags |= VEC_CLR;
            } else if (vop == VEC_NORM) {
                if (rnd() % 2) flags |= VEC_WIDE;
                if (rnd() % 2) flags |= VEC_GBROW;
                r.p0 = (flags & VEC_WIDE) ? (u32)((i32)rnd() >> 7) : (u32)((i32)rnd() >> 16);
                r.p2 = (1u << 30) + rnd() % ((1u << 30) + 1);
                r.p3 = (19 + rnd() % 25) | ((10 + rnd() % 7) << 8);
                r.faddr = rnd() % (FLB - 1000);
                for (u32 k = 0; k < 6 * r.n * r.rows; k++) fl[r.faddr + k] = (u8)rnd();
                for (u32 k = 0; k < r.n * r.rows; k++) fl[r.faddr + 6 * k + 5] = (rnd() & 1) ? 0xFF : 0x00, fl[r.faddr + 6 * k + 4] = (fl[r.faddr + 6 * k + 5] & 0xF0) | (rnd() & 15);
            } else if (vop == VEC_COPY && rnd() % 2) {                  // to int8: destination in bytes
                flags |= VEC_B8; r.p0 = rnd() % 16;
                r.dst = RAMB / 2 + rnd() % (RAMB / 2 - 400); r.dstr = 1 + rnd() % 3; r.drow = 40 + rnd() % 40;
                nb8++;
            } else if (vop == VEC_MUL) {
                r.p0 = 1 + rnd() % 20;
            } else if (vop == VEC_ROPE) {
                u32 str = 1 + rnd() % 3;
                r.src = NE / 2 + rnd() % (NE / 2 - 1200); r.sstr = 2 * str; r.srow = 36 * str; r.p1 = str; r.n = 16; r.rows = 1 + rnd() % 8;
                r.faddr = rnd() % (FLB - 100);
                for (u32 k = 0; k < 64; k++) fl[r.faddr + k] = (u8)rnd();
            } else if (vop == VEC_NSTAT) {
                // statistics of a fresh region, then the derived mean / reciprocal, then a norm that uses them
                r.sstr = 1 + rnd() % 3; r.p1 = 0;
                const u32 *rv0 = (const u32 *)&r;
                for (int k = 0; k < 13; k++) wr_reg(1, k, rv0[k]);
                top->go_v = 1; top->reg_wdata = VEC_STAT | VEC_CLR; tick(); top->go_v = 0; wait_done(false);
                memcpy((void *)VEC, &r, sizeof r); vec_go(VEC_STAT | VEC_CLR);
                r.p0 = r.n * r.rows; r.p1 = rnd() % 20000;
                for (int k = 0; k < 13; k++) wr_reg(1, k, rv0[k]);
                top->go_v = 1; top->reg_wdata = VEC_NSTAT; tick(); top->go_v = 0; wait_done(false);
                memcpy((void *)VEC, &r, 13 * 4); vec_go(VEC_NSTAT);
                if (top->o_mean != VEC->mean || top->o_rec != VEC->rec || top->o_sh1 != VEC->sh1) {
                    fprintf(stderr, "  nstat %x %x %u vs %x %x %u\n", top->o_mean, top->o_rec, top->o_sh1, VEC->mean, VEC->rec, VEC->sh1); bad++; }
                vop = VEC_NORM; flags = VEC_AUTO | (rnd() % 2 ? VEC_GBROW : 0);
                r.p3 = (10 + rnd() % 7) << 8; r.p0 = rnd(); r.p2 = rnd();
                r.faddr = rnd() % (FLB - 1000);
                for (u32 k = 0; k < 6 * r.n * r.rows; k++) fl[r.faddr + k] = (u8)rnd();
                for (u32 k = 0; k < r.n * r.rows; k++) fl[r.faddr + 6 * k + 5] = (rnd() & 1) ? 0xFF : 0x00, fl[r.faddr + 6 * k + 4] = (fl[r.faddr + 6 * k + 5] & 0xF0) | (rnd() & 15);
                vcnt[0]++;
            } else if (vop == VEC_SOFTMAX) {
                r.sstr = 1; r.srow = 30 + rnd() % 10;
                r.p0 = (10000 + rnd() % 8000) | ((20 + rnd() % 15) << 16);
                r.dst &= ~1u; r.dstr = 2; r.drow = 2 * r.n + 2 * (rnd() % 3);
                r.p2 = (NE - 64) & ~1u;
                // scores close together now and then, so that the exponent ROM is exercised
                if (rnd() % 2) for (u32 rr = 0; rr < r.rows; rr++) for (u32 k = 0; k < r.n; k++) {
                    u32 e = r.src + rr * r.srow + k; long long v = ((long long)(int)rnd() << (rnd() % 14));
                    put16(e, (int)(v & 0xFFFF)); put16(e + r.p1, (int)((v >> 16) & 0xFFFF)); put16(e + 2 * r.p1, (int)(v >> 32));
                    reload = true;
                }
            }
            if (reload) ram_load();
            const u32 *rv = (const u32 *)&r;
            for (int k = 0; k < 13; k++) wr_reg(1, k, rv[k]);
            u32 ctrl = vop | flags;
            top->go_v = 1; top->reg_wdata = ctrl; tick(); top->go_v = 0;
            wait_done(false);
            memcpy((void *)VEC, &r, 13 * 4); vec_go(ctrl);
            unsigned long long s = ((unsigned long long)VEC->s_hi << 32) | VEC->s_lo, q = ((unsigned long long)VEC->q_hi << 32) | VEC->q_lo;
            if (top->s_acc != s || top->q_acc != q) { fprintf(stderr, "  S/Q %llx %llx vs %llx %llx\n", (unsigned long long)top->s_acc, (unsigned long long)top->q_acc, s, q); bad++; }
            snprintf(what, sizeof what, "vec op %u flags %x n %u rows %u p0 %x p3 %x", vop, flags, r.n, r.rows, r.p0, r.p3);
            vels += (long long)r.n * r.rows; if (vop < 6) vcnt[vop]++; else vcnt[6]++;
        }
        bad += ram_compare();
        if (bad) { fails++; fprintf(stderr, "op %d FAILED: %s stall %d wide %d\n", op, what, stall_pct, wide_src); ram_load(); }
    }
    printf("LANES=%d: %d operations, %lld MAC steps (%d fan-in ops), %lld per-element steps, %llu clocks, failures %d\n", LANES, nops, macs, fcnt, vels, cyc, fails);
    printf("MAC operations per (weight source x output mode):");
    for (int a = 0; a < 4; a++) for (int b = 0; b < 4; b++) printf(" %d", cnt[a][b]);
    printf("\nper-element operations (nstat+auto-norm copy lut stat norm softmax rope/mul): %d %d %d %d %d %d %d\n", vcnt[0], vcnt[1], vcnt[2], vcnt[3], vcnt[4], vcnt[5], vcnt[6]);
    printf("of which: MAC with int8 x %d, copy to int8 %d, LUT asinh %d\n%s\n", nx8, nb8, nasinh, fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
