# Marigold architecture (iteration 1)

Owner: core agent. Status: fixed on 2026-11-15 for the overnight build. A change after this goes through a dated
entry in docs/decisions.md. Rules marked **proposed** in section 8 still need the coordinator's sign-off, because
the golden model has to follow them too.

Marigold is Poinsetta's always-on island with extra blocks (wake spotter, config memory, matcher, host link, GPIO)
plus a small core that is powered only between a wake and its message. RTL is under rtl/: reused Poinsetta blocks
keep their names (imh_*, rtl/PROVENANCE.md); new blocks are mg_*.

## 1. Block diagram

```
                 always-on island: clk_fe = 3.072 MHz, always powered                      core: clk = 98.304 MHz,
  pdm_dat_i ->+------------+  16 kHz  +-----------+                                       powered after a wake only
  pdm_clk_o <-| PDM front  |--------->| capture   |== segment samples (toggle CDC) ==>+---------------------------+
              | end (imh)  |  pcm     | ring 64 KB|<= ack ============================ | RV32IM controller (imh)   |
              +------------+    |     | 16 x 4 KB |                                   | PMEM 32 KB (boot copy)    |
                                |     | banks, PE |                                   | working SRAM 256 KB       |
                                v     +-----------+                                   | fan-out MAC array         |
                           +-------+    ^  ^ wake                                     |   L = 16 lanes, C <= 8    |
                           |  VAD  |----+  |                                          | MAC array + per-element   |
                           +-------+ vad,  |                                          |   unit, LANES = 8         |
                                |    raw   |                                          | weight ROM port (stall)   |
                                v          |                                          +-------------+-------------+
                      +-------------------+                                                         |
                      | wake spotter slot |<-- template, sens                       weight ROM <= 20 Mbit (fw +
                      | (rtl/wake, owned  |    (config read port)                   weights + tables, packed)
                      |  by the wake agt) |                                                       |
                      +-------------------+                                                       |
     scl/sda <-> +--------+   +------------+   +---------------+   island bridge (toggle CDC) <===+
                 | I2C    |-->| config     |-->| config SRAM   |<-- core reads config, writes q + values
                 | target |   | loader+CRC |   | 32 KB, kept   |
  fl_* <-------->| SPI    |-->|            |   | (arbiter)     |
                 | reader |   +------------+   +-------+-------+
                 +--------+                            |
                                            +----------v---------+   +-----------+
                                            | matcher (exact     |-->| message   |--> I2C read / UART tx, irq_o
                                            | rule, 1 MAC/clk)   |   | formatter |
                                            +--------------------+   +-----+-----+
                                                                           v
                                                                     +-----------+
                                                                     | GPIO rule |--> gpio_o[7:0]
                                                                     | engine    |
                                                                     +-----------+
                 always-on control: straps, reset, listen enable, core power and oscillator, isolation
```

## 2. Clocks, power domains, reset

- clk_fe, 3.072 MHz, always on (the microphone's PDM clock is clk_fe / 1, as on Poinsetta). Everything in the
  island runs on it, behind per-block clock gates (clk_gate.sv): the front end every clock, the VAD and capture
  ring once per sample, the rest only while they have work.
- clk, the core oscillator: 98.304 MHz nominal (32 x clk_fe; 45 nm blocks reached 200 to 240 MHz, so 28 nm has
  margin). Enabled by the island only between a wake and the core's power-off request.
- Domains: AON (island logic, 32 KB config SRAM, 8 GPIO pads), RING (16 capture banks of 4 KB, each with its own
  power enable), CORE (controller, PMEM, working SRAM, arrays, ROM; one switch, isolated outputs). The core loses
  all state when off and boots again on every wake: its program is copied from the ROM (Poinsetta's boot FSM,
  about 16 KB at 16 bytes per clock, under 15 us).
- In simulation, a powered-down memory (ring bank, PMEM, working SRAM) fills with random data, so a design that
  relies on lost state fails the bit-exact check (Poinsetta's method).
- rst_n: asynchronous assert, synchronous release per domain. The core is held in reset whenever it is off.

## 3. Always-on island

| Block | RTL | Notes |
|---|---|---|
| PDM front end | src/pdm_frontend.sv and children (reused as is) | 3.072 MHz one-bit stream to 16 kHz int16. Gain and DC blocker come from island registers (the core is off while listening). |
| VAD | src/vad.sv (reused as is) | vad_o (attack and 20-frame hangover) and raw_o (this frame alone looked like speech) per 256-sample frame. |
| Wake spotter slot | rtl/wake/ (wake agent) | Interface in section 3.1. Until it lands, mg_wake_stub fires on a test pin. |
| Capture ring | top/mg_capture.sv (from imh_capture.sv) | 2^15 samples = 64 KB in 16 banks of 4 KB with power enables. Segments open at a wake (rule in section 8.1), queue 2 deep, and are delivered to the core one sample per toggle handshake. |
| Config SRAM | top/mg_cfg_ram.sv | 32 KB, 32-bit words with byte enables, one port, fixed-priority arbiter with a grant (stall) for every master: loader write > wake spotter (real time) > GPIO > matcher > core bridge. |
| Config loader | top/mg_cfg_load.sv | Takes the blob as a byte stream from I2C (CFG_DATA), writes it to config SRAM, checks magic, version, field ranges, length and CRC-32. Only a blob that passes sets cfg_valid. Boot from SPI flash (spi_flash_reader.sv is in the tree) is cut from iteration 1 (coordinator scope ruling). |
| Matcher | top/mg_match.sv | The exact rule of docs/config-blob.md, steps 2 to 5, one int8 MAC per clk_fe clock straight from config SRAM (N x D clocks: a full 32 KB blob takes at most 10.7 ms). Inputs q (D <= 256 int8) and one raw value per config slot (level codes included), written by the core over the bridge before MATCH_GO; table fields after the vectors are read a byte at a time (they need not be word aligned). |
| Event and message formatter | top/mg_msg.sv | Events COMMAND, WAKE, NOMATCH, TIMEOUT (docs/config-blob.md, section 8.7). Builds 0xA5, type, action, n_values, {slot, i32 LE} per value, confidence, CRC-8 (section 8.4), at most 46 bytes, into a 2-frame FIFO; serves frames to I2C reads or sends them on the UART; drives irq_o. Event frames go to the host only when their flag bit is set. |
| I2C target | top/mg_i2c.sv | 7-bit address, standard and fast mode (100 and 400 kHz), sampled on clk_fe through two-flop synchronisers. No clock stretching. Register map in section 6. |
| UART | src/uart_tx.sv (reused) | 8N1, bit time 27 clk_fe clocks (113.8 kbaud, 1.2% from 115200). Used for messages when the host strap selects UART. |
| GPIO rule engine | top/mg_gpio.sv | 8 push-pull outputs, reset low. Applies the rules of the accepted action, or of event entry 0xF0 / 0xF1 / 0xF2 on WAKE / NOMATCH / TIMEOUT, whatever the flags (section 8.5); one pulse timer per pin. |
| Always-on control | top/marigold_top.sv | Straps, listen enable, core power and oscillator sequencing, isolation, the bridge's island side. |

### 3.1 Wake spotter interface (proposed to the wake agent; theirs replaces it)

```
module mg_wake #(...) (
    input  wire               clk, rst_n,      // clk_fe domain
    input  wire               en,              // listening (config valid and listen enabled)
    input  wire               in_valid,        // the PCM stream the capture ring sees, one sample per 192 clocks
    input  wire signed [15:0] in_data,
    input  wire               in_vad, in_raw,  // VAD flags as bundled with this sample (section 8.1)
    input  wire [14:0]        tpl_addr,        // byte address of the wake template in config SRAM
    input  wire [7:0]         tpl_len,
    input  wire [15:0]        sens,            // wake_sens from the blob
    output wire               cfg_req,         // config SRAM read port (arbiter; hold until cfg_gnt)
    output wire [14:0]        cfg_addr,
    input  wire               cfg_gnt,
    input  wire [31:0]        cfg_rdata,       // word at cfg_addr, valid the clock after the grant
    output wire               wake             // one clock, together with in_valid: wake decided at this sample
);
```

The capture ring records the index of the sample that carries `wake`; that index is the k of section 8.1.

## 4. Core (powered after a wake)

| Block | RTL | Size |
|---|---|---|
| Controller | core/imh_cpu.sv (reused) | RV32IM, clock gated while stalled. |
| Program memory | imh_pmem in core/imh_mem.sv | 32 KB, filled from the ROM by the boot FSM on every power-up. |
| Working SRAM | imh_ram | 256 KB (activations for a 2 to 3 s command). Every port has a stall. |
| Fan-out MAC array | core/imh_fan.sv | L = 16 lanes x up to C = 8 frames per weight read, int4 or int8 weights, per-frame clock gates. Half Poinsetta's 32 lanes. |
| MAC array + per-element unit | core/imh_mac.sv, imh_vec.sv | LANES = 8 (attention, norms, softmax, LUTs, requantisation). |
| Weight source | imh_wsrc_rom in core/imh_mem.sv | <= 20 Mbit ROM, one bit per cell, image = firmware + weights + tables (Poinsetta's mkimage style). Stall-tolerant stream port. |
| Island bridge | core/mg_core.sv | Core window 0x3000_0000 (section 5); each access is one toggle round trip (about 1 us). |

Sizing (estimate): 5M weights at 50 frames per second need 250M MACs per second in real time. At L = 16 and
98 MHz the array peaks at 1.57G MACs per second, and Poinsetta's measured 87% efficiency gives about 1.37G: 5.5
times real time. A streaming encoder (chunks of 80 ms, look-ahead of at most one chunk) therefore drains the 1 s
pre-roll in about 0.2 s, keeps up during the command, and has about 15 ms of encoder work left at the end of
speech (one chunk: 4 frames x 5M weights). The response time is then set by the end-of-speech rule (section 8.1,
128 ms at the default), the last chunk, matching (at most 10.7 ms) and the message. A non-streaming encoder over a
3 s segment (750M MACs) would take about 0.55 s after the end of speech and miss target 7, so **the intent
encoder must stream** at this array size (or the array doubles to L = 32).

## 5. Core memory map

| Address | Size | What |
|---|---|---|
| 0x0000_0000 | 32 KB | PMEM (program, copied from ROM at boot) |
| 0x1000_0000 | 256 KB | working SRAM |
| 0x2000_0000 | 4 KB | core registers: Poinsetta's map (rtl/core/fw/hal.h: UART slot unused, PCM delivery, weight-source seek, cycles, sleep) plus MAC 0x100, VEC 0x200, FAN 0x300 |
| 0x3000_0000 | 32 KB | config SRAM, read only from the core (island bridge) |
| 0x3000_8000 | 4 KB | island registers (island bridge), below |

Island registers (word offsets from 0x3000_8000):

| Offset | Name | Access | Meaning |
|---|---|---|---|
| 0x00 | ISL_ID | r | 0x4D474931 ("MGI1") |
| 0x04 | ISL_STATUS | r | [0] cfg_valid, [1] listening, [2] match busy, [3] message pending, [7:4] cfg error |
| 0x08 | MATCH_GO | w / r | w: 1 = run the matcher on Q and VAL (written first); r: [0] busy, [1] done, [2] rejected |
| 0x0C | MATCH_BEST | r | [15:0] best index b, [23:16] action[b], [31:24] the accepted action's slot_mask |
| 0x10 | MATCH_SB | r | s_b |
| 0x14 | MATCH_S2 | r | best score of a different action (0x8000_0000 when there is none) |
| 0x20 | SEG_PRE | r | pre-roll, samples (4800) |
| 0x24 | SEG_MIN | r | no end-of-speech close before this many samples (8000) |
| 0x28 | SEG_CAP | r | segment cap, samples (96000) |
| 0x2C | SEG_HANG | r | [7:0] hang frames (8), [15:8] onset frames (3) |
| 0x30 | SEG_LISTEN | r | listen window, samples (64000) |

The segment constants are fixed in iteration 1 (golden model parameters); conf_sh comes from the blob (byte 0x1C);
the PDM gain is 0 and the DC blocker on.
| 0x40 to 0x5C | VAL[0..7] | w | raw value of every config slot from the value head, int32 (level codes -1..-5) |
| 0x100 to 0x1FC | Q | w | the encoder output q, int8, byte d at 0x100 + d |

## 6. Pins and host interface

Pins (14 signal pads in iteration 1): clk_fe, rst_n, pdm_clk_o, pdm_dat_i, scl, sda (open drain), uart_tx_o,
irq_o, gpio_o[7:0]. The blob's host field picks where frames go (0 none, 1 I2C, 2 UART), so no host strap is needed.
SPI-flash config boot (4 more pads and a boot strap) is deferred to iteration 2.

I2C: the chip answers at 0x2A until a config is valid, then at the blob's i2c_addr (0x2A again if it is 0).
A write sets the register pointer with its first byte; further written bytes go to that register. A read
returns bytes from the pointer. The pointer increments after each byte except on the stream registers.

| Reg | Name | Access | Meaning |
|---|---|---|---|
| 0x00 | ID | r | 0x4D |
| 0x01 | STATUS | r | [0] cfg_valid, [1] loading or checking, [2] message ready, [3] listening, [7:4] cfg error (0 none, 1 magic, 2 version, 3 field range, 4 length, 5 CRC) |
| 0x02 | CTRL | w | [0] begin a config load (clears cfg_valid, stops listening), [1] commit (check, then listen if valid), [2] listen on, [3] drop the pending frames, [4] listen off |
| 0x03 | CFG_DATA | w, stream | next blob byte |
| 0x04 | MSG_LEN | r | bytes in the pending message (0 = none) |
| 0x05 | MSG_DATA | r, stream | next byte of the oldest pending frame; reading its last byte (the CRC) releases it; irq_o falls when no frame is left |
| 0x06, 0x07 | CFG_CNT | r | blob bytes received, low and high byte |
| 0x08 | EVT_STAT | r | [0] a frame was dropped because the 2-frame FIFO was full (cleared by this read) |

UART (strap_host = 1): each frame goes out as its bytes, 8N1; irq_o is high from a frame being ready to the
last stop bit of the last pending frame. GPIO rules run for every event whatever the host setting. irq_o is
used only when flags bit 0 is set; otherwise it stays low.

## 7. Command sequence

1. Reset: island up, core off. Config from I2C (CTRL begin, CFG_DATA bytes, CTRL commit) or from SPI flash.
   A valid blob sets cfg_valid; listening starts (front end, VAD, spotter; ring banks for the pre-roll window
   powered).
2. Listening: about one clk_fe-domain sample update per 192 clocks; only the front end runs every clock.
3. Wake at sample k: the ring opens a segment at max(k - pre, end of the last segment) and powers all its banks;
   the island raises a WAKE event (frame if flags bit 1, rules of entry 0xF0) and turns on the core (power,
   isolation off, oscillator on, reset released).
4. The core boots from ROM and reads the segment through the PCM port (catching up from the pre-roll). If the
   segment's end marker says TIMEOUT (no speech in the listen window, section 8.1), the island has already raised
   TIMEOUT and the core only requests power-off. Otherwise the core runs the encoder and the value head, writes Q and
   the raw value of every config slot (VAL) over the bridge, writes MATCH_GO, then requests power-off (Poinsetta's
   deep-sleep toggle). With no further segment queued, the island isolates and switches off the core.
5. The matcher runs from config SRAM (N x D clocks). Accept: COMMAND frame (always sent) and the action's rules.
   Reject: NOMATCH (frame if flags bit 2, rules of entry 0xF1).

## 8. Rules the golden model must share

### 8.1 Segment rule, listen window and TIMEOUT (proposed)
VAD frames are 256 samples from sample 0 after reset; the flags bundled with sample w (w a positive multiple of 256)
are frame f = w/256 - 1's vad and raw. A wake at sample k (the sample carrying `wake`) opens a segment, if none is
open, at start = max(k - pre, end of the previous segment). A wake while a segment is open does nothing.
At each sample w that is a positive multiple of 256 while the segment is open, with f = w/256 - 1:
- sp = the number of frames f' with raw = 1 and 256 f' >= k (frames that start at or after the wake sample);
- h = the number of consecutive frames ending at f with raw = 0 (counted from reset, not from the open).

Before speech (sp < onset), the segment closes at the first such w with w - k >= listen, as a TIMEOUT: its end
marker carries the timeout flag and the core runs no model. After speech (sp >= onset), it closes at the first such
w with w - start >= mn and h >= hang. In either state it also closes at the sample w with w - start = cap
(cap != 0). The end is exclusive. Defaults: pre 4800 (0.3 s), onset 3 frames, listen 64000 (4 s), mn 8000,
hang 8 frames (128 ms), cap 96000 (6 s). (Coordinator rulings of Nov 15: pre 4800, was 16000; mn 8000, was 32000.) raw is used rather than vad because vad_o has a 20-frame (320 ms)
hangover, too long for target 7.

### 8.2 Matching (docs/config-blob.md), two points made exact
- The margin test with no vector of a different action (every vector belongs to action[b]) passes: the maximum
  over an empty set is minus infinity.
- Scores are int32; with D <= 256 and |q|, |v| <= 128 they stay below 2^23, so no step can overflow.
- An accepted vector whose action id has no action-table entry gives NOMATCH.
- Values (ruling of Nov 15): the core writes one raw int32 per config slot. For each slot k of the accepted action
  (k < n_slots; 0 otherwise), codes -1 low, -2 medium, -3 high, -4 max, -5 min become the slot entry's level byte
  1, 2, 3 (a 0 byte means low = min, medium = floor((min + max) / 2), high = max) or the slot's max / min; then
  v = min(max(v, min), max). This is rtl/model/mg_model.py island_value and sw/intent/values.py island_value.

### 8.3 Confidence (proposed)
COMMAND and NOMATCH: confidence = clamp(s_b >> conf_sh, 0, 255) (arithmetic shift), s_b being the best score
(accepted or rejected), conf_sh from the blob (byte 0x1C; 9 at D = 128). WAKE and TIMEOUT: 0.

### 8.4 Message CRC-8 (proposed)
CRC-8 with polynomial 0x07, initial value 0x00, no reflection, no final XOR (the SMBus PEC), over every byte from
0xA5 to the confidence byte.

### 8.5 GPIO rules (proposed; the SDK writes these fields)
For an accepted action, rules gpio_first to gpio_first + gpio_count - 1 apply in order:
- mode 0 pulse: the pins in pin_mask go to bit 0 of `level`, then to the opposite level on the pulse_ms-th tick
  of a millisecond prescaler (3072 clk_fe clocks) that runs while any pulse runs: exactly pulse_ms ms for a pulse
  that starts the prescaler, within 1 ms otherwise (pulse_ms 0 or 1 ends on the first tick). A new pulse on a pin
  restarts its timer; any other mode on a pin cancels its pulse.
- mode 1 latch: the pins in pin_mask go to bit 0 of `level`.
- mode 2 toggle: the pins in pin_mask invert.
- mode 3 binary: the clamped value of slot `slot` goes out on the pins of pin_mask, least significant bit on the
  lowest pin; negative values give 0 and values above 2^k - 1 (k pins) give 2^k - 1. If the action does not carry
  that slot, the pins do not change.

### 8.7 Events
- WAKE: raised at the wake sample when a segment opens (not for a wake ignored during an open segment).
  Frame: type 1, action 0, no values, confidence 0.
- TIMEOUT: raised when a segment closes as a TIMEOUT (section 8.1). Type 3, action 0, no values, confidence 0.
- NOMATCH: the matcher rejected (step 4). Type 2, action 0, no values, confidence of section 8.3.
- COMMAND: accepted. Type 0, action[b], the action's slots in slot order (popcount of slot_mask), each value
  clamped to its slot's [min, max], confidence of section 8.3.
- Frames go into a 2-frame FIFO in event order; a frame that finds it full is dropped and EVT_STAT bit 0 set.
  COMMAND is always queued; WAKE, NOMATCH and TIMEOUT only when flags bits 1, 2, 3 are set. GPIO rules (the
  action's, or entry 0xF0 / 0xF1 / 0xF2 found by id in the action table; none if the entry is absent) apply for
  every event, whatever the flags.

### 8.6 Config checks
magic "MGC0", version 0, 1 <= n_actions <= 64, n_slots <= 8, n_gpio <= 32, host <= 2, D <= 256, total length
(0x20 + 4N + N D + 4 n_actions + 12 n_slots + 8 n_gpio + wake_len + 4) <= 32768 and equal to the bytes received,
then CRC-32 (IEEE: reflected polynomial 0xEDB88320, initial 0xFFFFFFFF, final XOR 0xFFFFFFFF, as zlib.crc32) over
every byte before the last four, compared with them.
