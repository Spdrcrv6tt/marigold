// Chip build of the hardware interface: register accesses. See hal.h.
#include "hal.h"
#define REG(o) (*(volatile u32 *)(MMIO_BASE + (o)))
extern char _end[];

void fl_seek(u32 a) { REG(REG_FL_SEEK) = a; }
u32 fl_u8(void) { return REG(REG_FL_DATA); }
void uart_putc(int c) { REG(REG_UART_TX) = (u32)c; }
void trace(u32 t, u32 v) { REG(REG_TRACE_TAG) = t; REG(REG_TRACE_VAL) = v; }
int hal_trace_on(void) { return (int)((REG(REG_GPIO) >> 2) & 1); }
u32 hal_heap_start(void) { return (u32)_end - RAM_BASE; }
u32 hal_heap_end(void) { return REG(REG_RAM_BYTES) - 4096; }      // the stack keeps the top 4 KB
// The controller sleeps (core clock stopped) whenever it waits for audio; between segments it sleeps deep (the
// capture buffer may stop the core oscillator). Audio comes from the capture buffer (rtl/top/imh_capture.sv): the
// samples of one segment in order, each with the segment flag, then an end marker. The buffer decides the segments:
// rec-gated, the stretch while rec is high; VAD-gated, VAD speech with PREROLL samples before its onset, split at
// the window cap (the next segment then continues at once), and it keeps recording while the core transcribes.
#define SLEEP(m) (REG(REG_SLEEP) = (m))
#define PCM_VALID 0x10000u
#define PCM_SEG   0x40000u
#ifndef STREAM_MAX_SEC
#define STREAM_MAX_SEC 20   // segment cap of the streaming firmware (stream.c)
#endif
#ifndef PREROLL
#define PREROLL 10240        // 640 ms (8 chunks): covers the VAD's measured onset delay (rtl/core/tools/vad_onset.py)
#endif
#ifndef SEG_MIN
#define SEG_MIN (63 * 1280)  // 5.04 s: no pause close before (format 3.4)
#endif
static int armed, vg;
static u32 cap_info;
u32 hal_cap_info(void) { return cap_info; }
u32 hal_capture(i16 *dst, u32 max, u32 *start) {
    u32 n = 0, d, g;
    *start = ~0u;
    REG(REG_GPIO) = 0;                                             // busy pin low while listening
    if (!armed) {                                                  // wait until rec arms the chip
        while (!((g = REG(REG_GPIO)) & 1)) SLEEP(8);               // deep: wake when rec != 0
        vg = (g >> 3) & 1;
        REG(REG_PCM_PRE) = vg ? PREROLL : 0;
        REG(REG_PCM_CAP) = vg ? max : 0;                           // rec-gated: one segment per press, cut at max here
        REG(REG_PCM_CTRL) = 1 | (1 << 4);                          // front end on, gain 0, DC blocker on
        armed = 1;
    }
    for (;;) {                                                     // the first sample of the next segment
        d = REG(REG_PCM_DATA);
        if (d & PCM_VALID) { if (d & PCM_SEG) break; continue; }
        g = REG(REG_GPIO);
        if (vg && !(g & 1)) { REG(REG_PCM_CTRL) = 0; armed = 0; return 0; }   // disarmed
        SLEEP(8 | ((g & 1) << 4));
    }
    *start = REG(REG_PCM_IDX);
    dst[n++] = (i16)d;
    for (;;) {
        d = REG(REG_PCM_DATA);
        if (d & PCM_VALID) { if (!(d & PCM_SEG)) break; if (n < max) dst[n++] = (i16)d; }
        else SLEEP(1);
    }
    cap_info = REG(REG_PCM_IDX) | ((d & 1) << 31);                 // end marker: high-water mark, overrun
    if (!vg) { REG(REG_PCM_CTRL) = 0; armed = 0; }                 // rec-gated: front end off between presses
    REG(REG_GPIO) = 1;                                             // busy: transcribing
    return n;
}
// streaming (stream.c): the same protocol one sample at a time
static u32 first_d;
static int have_first;
i32 hal_seg_begin(void) {
    u32 d, g;
    REG(REG_GPIO) = 0;
    if (!armed) {
        while (!((g = REG(REG_GPIO)) & 1)) SLEEP(8);
        vg = (g >> 3) & 1;
        REG(REG_PCM_PRE) = vg ? PREROLL : 0;                  // VAD-gated: golden format 3.4 (imh_capture.sv)
        REG(REG_PCM_CAP) = vg ? 16000u * STREAM_MAX_SEC : 0;  // 250 chunks
        REG(REG_PCM_MIN) = vg ? SEG_MIN : 0;
        REG(REG_PCM_CTRL) = 1 | (1 << 4);
        armed = 1;
    }
    for (;;) {
        d = REG(REG_PCM_DATA);
        if (d & PCM_VALID) { if (d & PCM_SEG) break; continue; }
        g = REG(REG_GPIO);
        if (vg && !(g & 1)) { REG(REG_PCM_CTRL) = 0; armed = 0; return -1; }
        SLEEP(8 | ((g & 1) << 4));
    }
    first_d = d; have_first = 1;
    REG(REG_GPIO) = 1;
    return (i32)REG(REG_PCM_IDX);
}
int hal_sample(i16 *v) {
    u32 d;
    if (have_first) { have_first = 0; *v = (i16)first_d; return 1; }
    for (;;) {
        d = REG(REG_PCM_DATA);
        if (d & PCM_VALID) break;
        SLEEP(1);
    }
    if (!(d & PCM_SEG)) {
        cap_info = REG(REG_PCM_IDX) | ((d & 1) << 31);
        if (!vg) { REG(REG_PCM_CTRL) = 0; armed = 0; }
        return 0;
    }
    *v = (i16)d; return 1;
}
u32 hal_backlog(void) { return REG(REG_PCM_BKL); }
void hal_fail(u32 code) { trace(0x00FF0000, code); for (;;) __asm__ volatile("ebreak"); }

void *memcpy(void *d, const void *s, unsigned n) { u8 *a = d; const u8 *b = s; while (n--) *a++ = *b++; return d; }
void *memset(void *d, int c, unsigned n) { u8 *a = d; while (n--) *a++ = (u8)c; return d; }

void main(void) {
    while (model_main()) ;
    REG(REG_GPIO) = 0;
}
