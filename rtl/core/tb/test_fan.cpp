// Unit test of imh_fan (fan-out array) against its C reference fan_go() in fw/hal_host.c: random operations over
// int8 / int4 weights, 1..8 frames, partial last groups, sat16, sat25 with and without accumulate, and argmax, with a
// weight source that offers a random number of bytes (0..16) each clock. After each operation the whole RAM and the
// argmax registers are compared; the clocks per operation are compared with the MAC count (efficiency).
//
//   test_fan [n_ops] [seed]        exit code 0 = all operations identical
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "Vfan_tb.h"
#include "verilated.h"
#include "hal.h"
void host_lib_init(u8 *r, u32 rbytes, u8 *f, u32 flen);
#define RAMB 131072
#define FLB 1048576
#define WBY (2 * LANES)

static Vfan_tb *top;
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
    if (!flav && (int)(rnd() % 100) >= stall_pct) flav = wide_src ? 1 + rnd() % LANES : LANES;
    top->clk = 0; top->eval();
    top->ws_avail = flav; memcpy(&top->ws_data, fl + (flptr % FLB), LANES);
    top->eval();
    cyc++;
}
static void wr_reg(int idx, u32 v) { top->reg_we = 1; top->reg_idx = idx; top->reg_wdata = v; tick(); top->reg_we = 0; }
static void ram_load() {
    for (u32 w = 0; w < RAMB / WBY; w++) { top->h_en = 1; top->h_we = 1; top->h_addr = w; memcpy(&top->h_wdata, mram + WBY * w, WBY); tick(); }
    top->h_en = 0; top->h_we = 0;
}
static int ram_compare() {
    int bad = 0;
    top->h_en = 1; top->h_we = 0; top->h_addr = 0; tick();
    for (u32 w = 0; w < RAMB / WBY; w++) {
        top->h_addr = w + 1;
        if (memcmp(&top->h_rdata, mram + WBY * w, WBY)) { if (!bad) fprintf(stderr, "  RAM word %u (element %u..) differs\n", w, w * LANES); bad++; }
        tick();
    }
    top->h_en = 0;
    return bad;
}

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    int nops = argc > 1 ? atoi(argv[1]) : 500;
    if (argc > 2) rs = (u32)atoi(argv[2]) | 1;
    int full = argc > 3;                     // "full": the weight source offers 16 bytes every clock
    int shapes = argc > 3 && !strcmp(argv[3], "shapes");   // Poinsetta's layer shapes, int4, efficiency per op
    static const u32 SH[][3] = {{320, 320, 8}, {320, 1280, 8}, {1280, 320, 8}, {1600, 640, 4}, {320, 320, 1}, {320, 2560, 1}, {1280, 320, 1}};
    if (shapes) nops = 7;
    top = new Vfan_tb;
    host_lib_init(mram, RAMB, fl, FLB);
    for (u32 i = 0; i < RAMB; i += 2) { i16 x = (rnd() % 10 == 0) ? ((rnd() & 1) ? 32767 : -32767) : (i16)rnd(); memcpy(mram + i, &x, 2); }
    for (u32 i = 0; i < FLB + 64; i++) fl[i] = (u8)rnd();
    top->clk = 0; top->rst_n = 0; top->reg_we = 0; top->go = 0; top->h_en = 0; top->h_we = 0; top->ws_avail = 0;
    for (int i = 0; i < 4; i++) tick();
    top->rst_n = 1; tick();
    ram_load();
    int fails = 0, cnt[4] = {0};
    unsigned long long macs = 0, clocks = 0;
    const u32 NE = RAMB / 2;
    for (int op = 0; op < nops && fails < 5; op++) {
        stall_pct = full ? 0 : (rnd() % 3 == 0) ? 0 : (int)(rnd() % 60);
        wide_src = full ? 0 : rnd() % 4 != 0;
        u32 w4 = shapes ? 1 : rnd() % 2, om = shapes ? 0 : rnd() % 3, resacc = 0, w16 = (!shapes && !w4 && rnd() % 2) ? 1 : 0;
        if (om == 2) om = 3;
        if (om == 1 && rnd() % 2) resacc = 1;
        fan_regs_t r; memset(&r, 0, sizeof r);
        r.nc = 1 + rnd() % FAN_CMAX;
        r.nin = 16 * (1 + rnd() % 12);
        r.nout = om == 3 ? 1 + rnd() % 150 : 1 + rnd() % 80;
        if (shapes) { r.nin = SH[op][0]; r.nout = SH[op][1]; r.nc = SH[op][2]; }
        r.xs = 16 * (r.nin / 16 + rnd() % 2);
        r.x = 16 * (rnd() % ((NE / 2 - r.nc * r.xs) / 16));
        u32 ng = (r.nout + FAN_L - 1) / FAN_L;
        r.ys = 16 * (ng * FAN_L / 16 + rnd() % 2);
        r.yp = om == 1 ? 16 * ((r.nc * r.ys) / 16 + rnd() % 2) : 0;
        u32 span = (r.nc - 1) * r.ys + ng * FAN_L + r.yp;
        r.y = NE / 2 + 16 * (rnd() % ((NE / 2 - span) / 16));
        u32 wbpi = w4 ? FAN_L / 2 : w16 ? 2 * FAN_L : FAN_L;       // weight bytes per input
        u32 site = ng * (8 * FAN_L + r.nin * wbpi);
        r.faddr = rnd() % (FLB - site);
        // header records: bias small enough, M and sh spread over the useful range
        for (u32 g = 0, p = r.faddr; g < ng; g++) {
            for (int l = 0; l < FAN_L; l++) {
                i64 b = (i64)(rnd() % 2000001) - 1000000; if (rnd() % 8 == 0) b <<= 12;
                for (int j = 0; j < 5; j++) fl[p + 8 * l + j] = (u8)(b >> (8 * j));
                u32 M = rnd() & 0xFFFF; fl[p + 8 * l + 5] = (u8)M; fl[p + 8 * l + 6] = (u8)(M >> 8);
                fl[p + 8 * l + 7] = (u8)(8 + rnd() % 30);
            }
            p += 8 * FAN_L + r.nin * wbpi;
        }
        u32 ctrl = w4 | (w16 << 1) | (om << 2) | (resacc << 6);
        // reference
        memcpy((void *)FAN, &r, sizeof r);
        fan_go(ctrl);
        // hardware
        u32 *rv = (u32 *)&r;
        for (int k = 0; k < 9; k++) wr_reg(k, rv[k]);
        top->go = 1; top->reg_wdata = ctrl; tick(); top->go = 0;
        unsigned long long c0 = cyc;
        while (!top->done) { tick(); if (cyc - c0 > 20000000) { fprintf(stderr, "timeout\n"); return 2; } }
        tick();
        clocks += cyc - c0; macs += (unsigned long long)r.nc * r.nin * ng * FAN_L;
        if (shapes) printf("  %4u x %4u, %u frames: %.1f MACs per clock (%.0f%%)\n", r.nout, r.nin, r.nc,
                           (double)r.nc * r.nin * ng * FAN_L / (cyc - c0), 100.0 * r.nc * r.nin * ng / (cyc - c0));
        int bad = ram_compare();
        if (om == 3 && (top->argidx != FAN->argidx || top->argval != FAN->argval)) {
            fprintf(stderr, "  argmax hw %u/%u ref %u/%u\n", top->argidx, top->argval, FAN->argidx, FAN->argval); bad++;
        }
        for (u32 c = 0; om == 3 && c < r.nc; c++) if (top->argidx_all[c] != FAN->argc[c]) {
            fprintf(stderr, "  argmax frame %u hw %u ref %u\n", c, top->argidx_all[c], FAN->argc[c]); bad++;
        }
        cnt[om]++;
        if (bad) {
            fails++;
            fprintf(stderr, "op %d FAIL: w4 %u om %u resacc %u nc %u nin %u nout %u x %u xs %u y %u ys %u yp %u\n",
                    op, w4, om, resacc, r.nc, r.nin, r.nout, r.x, r.xs, r.y, r.ys, r.yp);
            ram_load();
        }
    }
    printf("fan: %d ops (S16 %d, RES %d, ARGMAX %d), %d failures; %.1f MACs per clock (%.0f%% of 32), weight source stalls included\n",
           cnt[0] + cnt[1] + cnt[3], cnt[0], cnt[1], cnt[3], fails, (double)macs / clocks, 100.0 * macs / clocks / FAN_L);
    delete top;
    return fails ? 1 : 0;
}
