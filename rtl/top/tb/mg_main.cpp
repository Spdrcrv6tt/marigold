// Verilator harness for marigold_top (whole chip): clocks and reset, the microphone (PDM bits from a file, one per
// rising edge of pdm_clk_o), an I2C host that loads the config blob and reads every event frame when irq_o rises,
// a UART receiver (blobs with host = 2), the stand-in spotter's wake schedule, and loggers.
//
//   sim +pdm=<bits, one byte 0/1 each> +blob=<config blob> +out=<log> [+wakes=<file: one sample index per line>]
//       [+ratio=32] [+oscstart=2048] [+tail=1.0] [+max=<core steps>] [+pcmlog=<int16 file>] [+rawlog=<file>]
//       [+trace=1]
// Time steps are core clock periods; clk_fe (3.072 MHz) is one step in `ratio`, edges aligned. The core oscillator
// is modelled: clk ticks only while core_on is high, oscstart steps after it rose. After the PDM bits the microphone
// sends `tail` seconds of silence (alternating bits); the run ends when the core is off, no frame is pending and no
// GPIO pulse runs, or at +max.
// Log lines (sample = front-end samples delivered so far, the chip's own stream index):
//   S <status>                         I2C STATUS after the config load
//   F <step> <sample> <hex bytes>      an event frame read over I2C (irq) or received on the UART
//   G <step> <sample> <pins hex>       GPIO change
//   C <step> <sample> <0|1>            core switched off / on
//   E <tag hex> <val hex> <step>       firmware trace port
//   X <step> <sample>                  end of run
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include "Vmarigold_top.h"
#include "verilated.h"

static std::string arg(int argc, char **argv, const char *key, const char *def) {
    size_t n = strlen(key);
    for (int i = 1; i < argc; i++) if (!strncmp(argv[i], key, n) && argv[i][n] == '=') return std::string(argv[i] + n + 1);
    return def;
}
static std::vector<unsigned char> slurp(const std::string &p) {
    std::vector<unsigned char> v; FILE *f = fopen(p.c_str(), "rb");
    if (!f) { perror(p.c_str()); exit(2); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET); v.resize(n);
    if (fread(v.data(), 1, n, f) != (size_t)n) exit(2);
    fclose(f); return v;
}

static VerilatedContext *ctx;
static Vmarigold_top *top;
static FILE *lg, *pl, *rl;
static std::vector<unsigned char> pdm;
static std::vector<unsigned long long> wakes;
static size_t wk = 0, pk = 0, rises = 0, fe_n = 0, tailbits = 0;
static int ratio = 32, half = 16, osc_on = 0, prev_pclk = 0, prev_fe = 0, prev_on = 0, prev_gpio = 0, ubit = 27;
static unsigned long long cyc = 0, oscstart = 2048, osc_wait = 0, maxc = 0;
static int u_state = 0, u_cnt = 0, u_bits = 0, u_sh = 0;
static std::vector<int> ubytes;
static unsigned long long ustart = 0;
static int scl = 1, msda = 1;

// one front-end clock period
static void tick_fe() {
    for (int ph = 0; ph < ratio; ph++) {
        if (!top->core_on) { osc_on = 0; osc_wait = 0; }
        else if (!osc_on && ++osc_wait >= oscstart) osc_on = 1;
        top->scl_i = scl;
        top->sda_i = msda && !top->sda_oe;
        if (!osc_on && ph != 0 && ph != half) { cyc++; continue; }
        top->clk = osc_on; top->clk_fe = ph < half; top->eval();
        if (osc_on && top->trace_valid) {
            fprintf(lg, "E %08x %08x %llu\n", top->trace_tag, top->trace_val, cyc);
            if ((top->trace_tag >> 16) == 0xFF) { fprintf(stderr, "firmware failure code %u\n", top->trace_val); fflush(lg); exit(3); }
        }
        if (ph == 0) {
            int pc = top->pdm_clk_o;
            if (pc && !prev_pclk) { rises++; pk = rises - 1; }
            prev_pclk = pc;
            top->pdm_dat_i = (pk < pdm.size()) ? (pdm[pk] & 1) : (pk & 1);
            if (top->fe_valid_o && !prev_fe) {
                if (pl) { unsigned short w = top->fe_pcm_o; fwrite(&w, 2, 1, pl); }
                if (rl && fe_n > 0 && (fe_n & 255) == 0) fputc('0' + top->fe_raw_o, rl);
                fe_n++;
                while (wk < wakes.size() && wakes[wk] < fe_n) wk++;          // the stand-in spotter's next wake
                top->tst_wake_en = wk < wakes.size();
                top->tst_wake_at = wk < wakes.size() ? (unsigned)wakes[wk] : 0;
            }
            prev_fe = top->fe_valid_o;
            if (top->core_on != prev_on) { fprintf(lg, "C %llu %zu %d\n", cyc, fe_n, top->core_on); prev_on = top->core_on; }
            if (top->gpio_o != prev_gpio) { fprintf(lg, "G %llu %zu %02x\n", cyc, fe_n, top->gpio_o); prev_gpio = top->gpio_o; }
            int tx = top->uart_tx_o;                                          // UART receiver, 8N1
            if (u_state == 0) { if (!tx) { u_state = 1; u_cnt = ubit + ubit / 2; u_bits = 0; u_sh = 0; } }
            else if (--u_cnt == 0) {
                if (u_bits < 8) { u_sh |= tx << u_bits; u_bits++; u_cnt = ubit; }
                else {
                    if (ubytes.empty()) ustart = cyc;
                    ubytes.push_back(u_sh); u_state = 0;
                    int n = ubytes.size();                                    // a whole frame: 6 + 5 n_values bytes
                    if (n >= 4 && n == 6 + 5 * ubytes[3]) {
                        fprintf(lg, "F %llu %zu", cyc, fe_n);
                        for (int b : ubytes) fprintf(lg, " %02x", b);
                        fprintf(lg, "\n"); ubytes.clear();
                    }
                }
            }
        }
        top->clk = 0; top->eval();
        cyc++;
    }
}

// ---- I2C host, fast mode (5 clk_fe low, 3 high: 384 kHz)
static void wait_fe(int n) { for (int i = 0; i < n; i++) tick_fe(); }
static int i2c_bit(int b) {
    msda = b; wait_fe(5); scl = 1; wait_fe(1);
    int v = msda && !top->sda_oe;
    wait_fe(2); scl = 0;
    return v;
}
static void i2c_start() { msda = 1; wait_fe(5); scl = 1; wait_fe(3); msda = 0; wait_fe(3); scl = 0; }
static void i2c_stop() { msda = 0; wait_fe(5); scl = 1; wait_fe(3); msda = 1; wait_fe(8); }
static int i2c_wbyte(int x) { for (int i = 7; i >= 0; i--) i2c_bit((x >> i) & 1); return i2c_bit(1) == 0; }
static int i2c_rbyte(int ack) { int v = 0; for (int i = 0; i < 8; i++) v = (v << 1) | i2c_bit(1); i2c_bit(ack ? 0 : 1); return v; }
static int i2c_write(int dev, int reg, const unsigned char *d, size_t n) {
    i2c_start();
    int ok = i2c_wbyte(dev << 1) && i2c_wbyte(reg);
    for (size_t i = 0; ok && i < n; i++) ok = i2c_wbyte(d[i]);
    i2c_stop();
    return ok;
}
static int i2c_read(int dev, int reg, int n, std::vector<int> &out) {
    i2c_start();
    int ok = i2c_wbyte(dev << 1) && i2c_wbyte(reg);
    if (ok) { i2c_start(); ok = i2c_wbyte((dev << 1) | 1); }
    out.clear();
    for (int i = 0; ok && i < n; i++) out.push_back(i2c_rbyte(i < n - 1));
    i2c_stop();
    return ok;
}

int main(int argc, char **argv) {
    ctx = new VerilatedContext; ctx->commandArgs(argc, argv);
    top = new Vmarigold_top(ctx);
    pdm = slurp(arg(argc, argv, "+pdm", ""));
    std::vector<unsigned char> blob = slurp(arg(argc, argv, "+blob", ""));
    lg = fopen(arg(argc, argv, "+out", "mg.log").c_str(), "w");
    std::string pf = arg(argc, argv, "+pcmlog", ""), rf = arg(argc, argv, "+rawlog", ""), wf = arg(argc, argv, "+wakes", "");
    pl = pf.empty() ? 0 : fopen(pf.c_str(), "wb");
    rl = rf.empty() ? 0 : fopen(rf.c_str(), "w");
    if (!wf.empty()) { FILE *f = fopen(wf.c_str(), "r"); unsigned long long w; while (f && fscanf(f, "%llu", &w) == 1) wakes.push_back(w); if (f) fclose(f); }
    std::sort(wakes.begin(), wakes.end());
    ratio = atoi(arg(argc, argv, "+ratio", "32").c_str()); half = ratio / 2;
    oscstart = strtoull(arg(argc, argv, "+oscstart", "2048").c_str(), 0, 0);
    tailbits = (size_t)(atof(arg(argc, argv, "+tail", "1.0").c_str()) * 3072000.0);
    maxc = strtoull(arg(argc, argv, "+max", "100000000000").c_str(), 0, 0);
    top->trace_en_i = atoi(arg(argc, argv, "+trace", "1").c_str());
    top->tst_wake_en = !wakes.empty(); top->tst_wake_at = wakes.empty() ? 0 : (unsigned)wakes[0];
    top->scl_i = 1; top->sda_i = 1; top->pdm_dat_i = 0;
    // power-on reset (as Poinsetta): high, low, high, so every asynchronous reset sees an edge
    top->rst_n = 1;
    for (int i = 0; i < 8 * ratio; i++) { top->clk = !top->clk; top->clk_fe = (i / ratio) & 1; top->eval(); }
    top->rst_n = 0;
    for (int i = 0; i < 8 * ratio; i++) { top->clk = !top->clk; top->clk_fe = (i / ratio) & 1; top->eval(); }
    top->clk = 0; top->clk_fe = 0; top->eval();
    top->rst_n = 1;
    wait_fe(16);
    // config load over I2C at the default address, then STATUS at the blob's address
    int dev0 = 0x2A, dev = blob.size() > 15 && (blob[15] & 0x7F) ? (blob[15] & 0x7F) : 0x2A;
    int host = blob.size() > 14 ? blob[14] : 0;
    unsigned char c1 = 1, c2 = 2;
    if (!i2c_write(dev0, 0x02, &c1, 1) || !i2c_write(dev0, 0x03, blob.data(), blob.size()) || !i2c_write(dev0, 0x02, &c2, 1)) {
        fprintf(stderr, "config load: no ACK\n"); return 4;
    }
    std::vector<int> r;
    int okv = i2c_read(dev, 0x01, 1, r);
    if (!okv) { okv = i2c_read(dev0, 0x01, 1, r); dev = dev0; }
    fprintf(lg, "S %02x\n", okv ? r[0] : 0xFFF); fflush(lg);
    fprintf(stderr, "config: %zu bytes, status %02x, host %d, address %02x\n", blob.size(), okv ? r[0] : 0xFFF, host, dev);
    // stream the microphone; read frames when irq_o rises (host = 1)
    while (cyc < maxc && !ctx->gotFinish()) {
        tick_fe();
        if (host == 1 && top->irq_o) {
            std::vector<int> len, fr;
            if (i2c_read(dev, 0x04, 1, len) && len[0] > 0 && i2c_read(dev, 0x05, len[0], fr)) {
                fprintf(lg, "F %llu %zu", cyc, fe_n);
                for (int b : fr) fprintf(lg, " %02x", b);
                fprintf(lg, "\n"); fflush(lg);
            }
        }
        if (pk >= pdm.size() + tailbits && !top->core_on && !top->irq_o && u_state == 0 && ubytes.empty()) {
            wait_fe(3072 * 20);                                               // let GPIO pulses (<= 20 ms here) end
            if (!top->core_on && !top->irq_o) break;
        }
    }
    fprintf(lg, "X %llu %zu\n", cyc, fe_n);
    fprintf(stderr, "steps %llu samples %zu\n", cyc, fe_n);
    fclose(lg); if (pl) fclose(pl); if (rl) fclose(rl);
    top->final();
    return 0;
}
