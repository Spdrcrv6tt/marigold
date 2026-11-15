// Imhotep controller firmware: hardware interface. One C source (model.c) runs on the chip (hal_rv.c)
// and on the PC (hal_host.c, where the MAC array and the per-element unit are C functions).
#ifndef HAL_H
#define HAL_H

typedef signed char i8;
typedef unsigned char u8;
typedef short i16;
typedef unsigned short u16;
typedef int i32;
typedef unsigned u32;
typedef long long i64;
typedef unsigned long long u64;

#ifndef LANES
#define LANES 16             // must equal the LANES parameter of imh_core (power of two, >= 4)
#endif

// ---- memory map (chip)
#define PMEM_BASE 0x00000000u
#define RAM_BASE  0x10000000u
#define MMIO_BASE 0x20000000u

// ---- MAC array registers (MMIO_BASE + 0x100). Addresses x, y and their strides are in 16-bit elements
// from the start of the activation RAM; w, h and the w strides are in bytes. Writing ctrl runs the operation;
// the write returns when it is done. Semantics: rtl/core/imh_mac.sv, reference mac_go() in hal_host.c.
typedef struct {
    u32 faddr;     // 00 byte address of the weight stream (stream sources)
    u32 x;         // 04 first input element
    u32 xi;        // 08 input stride per i (fan-in index)
    u32 xo;        // 0c input stride per o (output row)
    u32 w;         // 10 first weight, RAM byte address (RAM sources)
    u32 wi;        // 14 weight stride per i, bytes
    u32 wo;        // 18 weight stride per o, bytes
    u32 y;         // 1c first output element
    u32 ys;        // 20 output stride per o
    u32 yp;        // 24 element offset between output planes
    u32 nin;       // 28 fan-in
    u32 nout;      // 2c rows
    u32 ctrl;      // 30 see below
    u32 argidx;    // 34 (read) argmax row of the last MAC_O_ARGMAX operation
    u32 argval;    // 38 (read) low 32 bits of its value
    u32 satflag;   // 3c (read) sticky: an accumulator saturated since reset (must stay 0)
    u32 h;         // 40 MAC_HDRRAM: RAM byte address of the per-row multipliers (u32 each)
    u32 hsh;       // 44 MAC_HDRRAM: right shift
    u32 xw;        // 48 ring: x offset of input i in row o is (xs0 + i*xi) mod xw (0 = no ring); xi, xs0 < xw
    u32 xs0;       // 4c
} mac_regs_t;

#define MAC_W_FLASH8 0u          // int8 from the weight source (ROM or flash), row header first
#define MAC_W_FLASH4 1u          // int4, two per byte
#define MAC_W_RAM16  2u
#define MAC_W_RAM32  3u
#define MAC_O_S16    (0u << 2)   // sat16(rshift_round(acc * M, sh)), one plane
#define MAC_O_RES    (1u << 2)   // sat to +-(2^24 - 1), two planes: low 16 bits, high 16 bits
#define MAC_O_RAW    (2u << 2)   // 48-bit accumulator, three planes
#define MAC_O_ARGMAX (3u << 2)   // rshift_round(acc * M, sh) unsaturated; running argmax, lowest index wins
#define MAC_SCALAR   (1u << 4)   // lane 0 only, x element addressed, one weight per clock
#define MAC_FANIN    (1u << 5)   // x contiguous and word aligned, weights across the lanes, one output element
#define MAC_RESACC   (1u << 6)   // with MAC_O_RES: out = sat(old + result), old read from the output planes
#define MAC_HDRRAM   (1u << 7)   // RAM weights: per-row M from RAM (h), sh = hsh
#define MAC_X8       (1u << 8)   // x is int8, addressed in bytes (batch: LANES bytes, x a multiple of LANES)

// ---- per-element unit registers (MMIO_BASE + 0x200). Two-dimensional loops over 16-bit elements:
// source src + r*srow + i*sstr, destination dst + r*drow + i*dstr, r < rows, i < n.
typedef struct {
    u32 src, sstr, srow;   // 00 04 08
    u32 dst, dstr, drow;   // 0c 10 14
    u32 n, rows;           // 18 1c
    u32 p0, p1, p2, p3;    // 20 24 28 2c  operation parameters
    u32 faddr;             // 30 weight-source byte address of the G/B records (NORM)
    u32 ctrl;              // 34 write = run
    u32 s_lo, s_hi;        // 38 3c (read) S accumulator
    u32 q_lo, q_hi;        // 40 44 (read) Q accumulator
    u32 mean, rec, sh1;    // 48 4c 50 (read) results of the last VEC_NSTAT
} vec_regs_t;

#define VEC_MUL     0u           // dst = sat16(rshift_round(a * b, p0)), b at src + p1
#define VEC_COPY    1u           // dst = src
#define VEC_LUT     2u           // dst = lut(src); p0 = f | f_out << 4 | table << 8 | tails << 10 | sat << 11 | acc << 12
#define VEC_STAT    3u           // S += x, Q += x * x
#define VEC_NORM    4u           // p0 = mean, p2 = rec, p3 = sh1 | sh2 << 8; G/B from the weight source at faddr
#define VEC_SOFTMAX 5u           // p0 = C | sh << 16, p1 = plane offset, p2 = element address of the reciprocal table
#define VEC_NSTAT   6u           // norm statistics from S, Q: p0 = count, p1 = eps -> mean, rec, sh1
#define VEC_ROPE    7u           // rotate pairs (x, x + p1) by the cos/sin row at faddr; n = 16 pairs, rows = heads
#define VEC_CLR     (1u << 3)    // clear S and Q first
#define VEC_WIDE    (1u << 4)    // source is 32 bit in two planes (low at src, high at src + p1)
#define VEC_B8      (1u << 7)    // COPY to int8: dst byte = sat(rshift_round(src, p0), 127), dst in bytes
#define VEC_GBROW   (1u << 5)    // NORM: one G/B record per row instead of per element
#define VEC_AUTO    (1u << 6)    // NORM: mean, rec, sh1 from the last VEC_NSTAT (p3 = sh2 << 8 only)

// ---- other registers (chip), MMIO_BASE + ...
#define REG_UART_TX   0x00   // w: byte (stalls while the transmitter is busy)
#define REG_UART_DIV  0x04   // rw: bit time = div + 1 clocks
#define REG_PCM_CTRL  0x08   // rw: [0] front end enable, [3:1] gain, [4] DC blocker
#define REG_PCM_DATA  0x0c   // r: [15:0] sample, [16] valid (the read takes it), [17] overrun (sticky), [18] segment:
                             //    1 = a sample of the segment, 0 = end marker (then [0] = capture-buffer overrun and
                             //    REG_PCM_IDX = its high-water mark in samples); rtl/top/imh_capture.sv
#define REG_GPIO      0x10   // r: [0] rec pin, [1] vad, [2] trace strap, [3] VAD-gate strap; w: [0] busy pin
#define REG_FL_SEEK   0x14   // w: weight-source byte address, starts a sequential read
#define REG_FL_DATA   0x18   // r: next byte (stalls until it is there)
#define REG_TRACE_TAG 0x1c   // w: tag latch
#define REG_TRACE_VAL 0x20   // w: value; pulses the trace port {tag, value}
#define REG_RAM_BYTES 0x24   // r: size of the activation RAM
#define REG_CYCLES    0x28   // r: clock counter, low 32 bits
#define REG_FL_CFG    0x2c   // rw: [2:0] flash read mode, [7:3] dummy cycles, [9:8] rx delay
#define REG_PCM_IDX   0x30   // r: stream index of the sample the last REG_PCM_DATA read returned
#define REG_SLEEP     0x34   // w: stop the core clock until [0] a sample arrives, [1] a sample of a segment
                             //    arrives, [2] the rec pin changes; [3] deep sleep: the capture buffer may stop
                             //    the core oscillator and wakes the core when a segment is ready or rec != [4]
#define REG_PCM_PRE   0x38   // rw: pre-roll of VAD-gated segments, samples (write while the front end is off)
#define REG_PCM_CAP   0x3c   // rw: window cap, samples (0 = none); a longer stretch continues in the next segment
#define REG_PCM_BKL   0x40   // r: capture backlog (samples written, not yet delivered) when the sample the last
                             //    REG_PCM_DATA read returned was presented
#define REG_PCM_MIN   0x44   // rw: VAD-gated segments: no pause close before a segment holds this many samples
#define REG_MAC       0x100
#define REG_VEC       0x200
#define REG_FAN       0x300

// ---- fan-out array (Poinsetta, rtl/core/imh_fan.sv): FAN_L output rows per group, up to FAN_CMAX frames per weight
#define FAN_L        32
#define FAN_CMAX     8
typedef struct {
    u32 faddr;     // 0  weight-source byte address of the site (group-interleaved layout, imh_fan.sv)
    u32 x;         // 4  element address of x[0][0]
    u32 xs;        // 8  frame stride of x, elements
    u32 nin;       // 12 inputs (multiple of 16)
    u32 nout;      // 16 output rows
    u32 nc;        // 20 frames, 1..FAN_CMAX
    u32 y;         // 24 element address of y[0][0]
    u32 ys;        // 28 frame stride of y, elements
    u32 yp;        // 32 RES: plane offset, elements
    u32 pad[3];
    u32 ctrl;      // 48 write: start (FAN_W4 | FAN_O_* | FAN_RESACC), stalls until done
    u32 argidx;    // 52 ARGMAX result
    u32 argval;    // 56
    u32 pad2;
    u32 argc[FAN_CMAX]; // 64 ARGMAX index of every frame (frame 0 also in argidx)
} fan_regs_t;
#define FAN_W4       1u          // int4 weights (else int8)
#define FAN_W16      2u          // int16 weights
#define FAN_O_S16    (0u << 2)
#define FAN_O_RES    (1u << 2)
#define FAN_O_ARGMAX (3u << 2)   // per frame: argc[c]; frame 0 also argidx, argval
#define FAN_RESACC   (1u << 6)

// ---- what model.c uses
#ifdef HOST
extern u8 *ram;                      // activation RAM
extern volatile mac_regs_t *MAC;
extern volatile vec_regs_t *VEC;
extern volatile fan_regs_t *FAN;
void mac_go(u32 ctrl);
void vec_go(u32 ctrl);
void fan_go(u32 ctrl);
#define FAN_GO(c) fan_go(c)
#define MAC_GO(c) mac_go(c)
#define VEC_GO(c) vec_go(c)
#else
#define ram ((u8 *)RAM_BASE)
#define MAC ((volatile mac_regs_t *)(MMIO_BASE + REG_MAC))
#define VEC ((volatile vec_regs_t *)(MMIO_BASE + REG_VEC))
#define FAN ((volatile fan_regs_t *)(MMIO_BASE + REG_FAN))
#define FAN_GO(c) (FAN->ctrl = (c))
#define MAC_GO(c) (MAC->ctrl = (c))
#define VEC_GO(c) (VEC->ctrl = (c))
#endif
void fl_seek(u32 addr);
u32  fl_u8(void);
void uart_putc(int c);
void trace(u32 tag, u32 val);
int  hal_trace_on(void);
u32  hal_heap_start(void);           // first free byte offset in the activation RAM
u32  hal_heap_end(void);             // one past the last byte the allocator may use
u32  hal_capture(i16 *dst, u32 max, u32 *start); // one utterance (rec pin, or VAD with the gate strap): sample count,
                                                  // *start = stream index of its first sample; ~0u = no more audio
u32  hal_cap_info(void);             // capture buffer after the last segment: high-water mark (samples) | overrun << 31
i32  hal_seg_begin(void);            // streaming: wait for the next segment; its first stream index, or -1 (no more audio)
int  hal_sample(i16 *v);             // the next sample of the segment (1), or 0 at its end (then hal_cap_info())
u32  hal_backlog(void);              // capture backlog (samples) when the last sample hal_sample returned was presented
void hal_fail(u32 code);

int  model_main(void);               // one utterance; 0 when there is no more audio

#endif
