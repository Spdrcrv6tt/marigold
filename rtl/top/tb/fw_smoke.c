// Smoke-test firmware for the whole-chip harness (no model): takes every sample of a segment, and unless the end
// marker says TIMEOUT writes a q and slot values derived from the segment's sample sum to the island, starts the
// matcher, waits for it, and asks to be switched off. rtl/top/tb/mg_smoke.py computes the same q in Python.
typedef unsigned int u32;
typedef int i32;
#define MMIO 0x20000000u
#define ISL  0x30000000u
#define R(a) (*(volatile u32 *)(a))
int main(void) {
    u32 n = 0, sum = 0;
    for (;;) {
        u32 d = R(MMIO + 0x0c);                                   // REG_PCM_DATA: [16] valid, [18] segment sample
        if (!(d & (1u << 16))) { R(MMIO + 0x34) = 1; continue; }  // sleep until a sample arrives
        if (d & (1u << 18)) { n++; sum += (u32)(i32)(short)(d & 0xFFFF); continue; }
        if (!(d & 2)) {                                           // end marker, not a TIMEOUT
            for (u32 w = 0; w < 64; w++) {
                u32 x = 0;
                for (u32 b = 0; b < 4; b++) { u32 k = 4 * w + b; x |= (((sum >> (k & 15)) ^ (k * 37u) ^ n) & 0xFFu) << (8 * b); }
                R(ISL + 0x8100 + 4 * w) = x;
            }
            for (u32 k = 0; k < 8; k++) R(ISL + 0x8040 + 4 * k) = (u32)((i32)(sum & 0x3FF) - 512 + (i32)k);
            R(ISL + 0x8008) = 1;
            while (!(R(ISL + 0x8008) & 2)) ;
        }
        n = 0; sum = 0;
        R(MMIO + 0x34) = 8;                                       // deep sleep: switched off
    }
}
