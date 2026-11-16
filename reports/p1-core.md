# Marigold core and chip RTL: iteration 1 (core agent)

Nov 15 to 16, 2026. The RTL is in rtl/ (provenance: rtl/PROVENANCE.md). The design is in docs/architecture.md. The
golden models are by the model agent (sw/golden, docs/golden-format.md v1.0) and the wake agent (sw/wake/golden,
docs/golden-wake.md format 2). Simulated numbers come from Verilator 5.054 and cocotb 2.1.0 in WSL. Area and power
are **estimates**, built on Poinsetta's method and densities, and are labelled as such.

## 1. Results against the iteration-1 targets this track owns

| # | Target | Result | How measured |
|---|---|---|---|
| 12 | Bit-exact hardware, mic pin to output pins, with negative controls | **Met on qsmoke** (golden format v1.0, placeholder weights): every frame, q vector, raw slot value, settled GPIO state and front-end sample is identical; the three negative controls fail. v1 rerun: section 9 | rtl/top/tb/mg_e2e.py: Verilator whole chip against sw/golden/chip_golden.py |
| 7 | Response time <= 300 ms median | **156 and 163 ms** from end of speech to irq in RTL (2 commands): 128 ms end-of-speech hangover (rule 8.1, hang 8 frames) + up to 16 ms of frame alignment + 21 to 35 ms of core and matcher work after the segment closes | the same runs; end of speech = end of the last VAD raw-speech frame |
| 8 | <= 5M weights at 4-bit, <= 20 Mbit ROM as built | 4,586,560 weights; ROM image **19.33 Mbit** (2,415,680 B: firmware 14,376 B, tables and weights from 0x8040) | rtl/core/tools/mkimage_mg.py image, the bytes the RTL's ROM holds |
| 9 | Listening power <= 100 uW at 28 nm | **20 to 103 uW (estimate)**: met unless the PDM front end lands at the top of Poinsetta's range | section 6 |

## 2. What was built

- **Island** (clk_fe 3.072 MHz, always on), docs/architecture.md section 3:
  - reused from Poinsetta 1719528 unchanged: PDM front end, VAD;
  - wake-gated capture ring (top/mg_capture.sv: 64 KB in 16 banks with power enables);
  - config loader with CRC-32 (mg_cfg_load) and 32 KB config SRAM with a stall-tolerant arbiter (mg_cfg_ram);
  - matcher with the exact rule, level codes and clamping (mg_match);
  - event frames with CRC-8, a 2-frame FIFO, I2C/UART and irq (mg_msg);
  - GPIO rule engine (mg_gpio), I2C target (mg_i2c), island control and bridge (mg_ctl);
  - the wake agent's spotter v2 (rtl/wake, VAD-gated).
- **Core** (98.304 MHz, power-gated between commands): Poinsetta's RV32IM controller, MAC array, per-element unit and
  fan-out array, with FAN_L = 16 (half of Poinsetta's 32), a 256 KB working SRAM and 32 KB PMEM booted from ROM on
  every wake. core/mg_core.sv derives from imh_core.sv, with no UART and an island bridge at 0x3000_0000.
- **Firmware** rtl/core/fw/marigold.c (helper agent, from Poinsetta's stream.c): streaming encoder, pooled head,
  CTC value head, writes of q and raw values, power-off. It matches the golden model bit for bit on the PC build
  (17 of 17 runs per export). Peak RAM is 175 KB of 256 KB; .text is 14.4 KB of 32 KB.
- **Top** rtl/top/marigold_top.sv. **Harness** rtl/top/tb/mg_main.cpp: PDM bits in, the config loaded over I2C,
  frames read on irq, a UART receiver, logs of GPIO, core power and bridge writes.

## 3. Verification

| Level | Test | Result |
|---|---|---|
| Reused blocks | rtl/tb cocotb (PDM 3 variants, VAD on LibriSpeech, SPI flash 4, UART, MAC lane 4, RAM 2) | 53 of 53 pass |
| Core units | test-fan (FAN_L 16), test-mac (16, 8, 4 lanes + streaming image, the export's LUT tables) | 467 and 24,000 operations, 0 failures; fan-out array 14.3 of 16 MACs per clock at full weight rate |
| Config + matcher | test_cfg.py: 440 loads, every corruption class; 150 random configs, a stall-injecting reader | all identical to rtl/model/mg_model.py (98 accepts, 52 rejects, 27 with no other action) |
| Island control | test_ctl.py: I2C at fast-mode minimum timing, config over I2C, 64 random events, frames over I2C and UART, GPIO | 35 frames identical, GPIO equal |
| Capture ring | test_cap.py: random wakes, speech and silence | 7 segments (4 TIMEOUT), 8 WAKE events, 24,530 delivered samples identical |
| Wake spotter | the wake agent's rtl/wake/tb (v2 plain and stall) | bit-exact per its report (reports/p1-wake.md) |
| Whole chip, smoke firmware | 3 wakes, three core power cycles | 6 frames identical |
| Whole chip, qsmoke | LibriSpeech, stand-in wakes: 3 segments | frames (6), q, values, GPIO, front end identical |
| Whole chip, qsmoke | "Hey Breeze, turn the fan up a little" x2, spotter v2, fan.mgc with host I2C | frames (4), q, values, GPIO, front end identical |
| Negative controls | ROM weight bit; mic bits inside a segment; config CRC | all fail as required (q differs; q and frames differ; STATUS 0x50 and no frames) |

The whole-chip runs found one real bug, now fixed. When the core clock stopped before reset reached the core
domain, the core kept its state through power-off, and the sample handshake fell out of sync on the next wake.
The core reset now asserts as soon as core_on falls. Two known gaps remain:
- The TIMEOUT path is covered by the capture and island-control unit tests, not by a whole-chip run.
- The ROM negative control changed q but no frame: with placeholder weights that one bit moved no decision.

## 4. Sizing of the core

- 16 lanes at 98 MHz run Poinsetta's measured efficiency (14.3 of 16 MACs per clock). The model needs 5.6M MACs per
  frame at 50 frames/s, so the core keeps up with streaming speech with margin (the model agent puts it at about
  4.9x real time).
- The measured 21 to 35 ms from segment close to irq covers the last chunk, the head, the value head, the island
  writes and the matcher.

## 5. Area at 28 nm (estimate)

Logic: Yosys with the NanGate45 typical library (rtl/syn/area.sh), memories removed. The cell area is converted to
NAND2 equivalents (0.798 um2 each) and placed at Poinsetta's 28 nm logic density (0.50 um2 per NAND2 at 70%).
ROM and SRAM use Poinsetta's 28 nm densities: ROM 0.06 um2 per bit at 70% array efficiency, SRAM 0.12 um2 per bit
at 60%.

| Block | Basis | mm2 |
|---|---|---|
| Core logic (controller, MAC array, per-element unit, fan-out array, LUT ROM as logic) | 262,552 um2 at 45 nm, 329k NAND2 | 0.235 |
| Island control (config, matcher, frames, GPIO, I2C) | 20,341 um2, 25k NAND2 | 0.018 |
| Capture ring logic | 36,806 um2 (upper bound; ring shrunk for synthesis) | 0.033 |
| PDM front end + VAD | 7,032 um2 | 0.006 |
| Wake spotter v2 with its memories and ROM | wake agent's estimate | 0.15 to 0.18 |
| Weight ROM, 19.33 Mbit as built | 0.0857 um2 per bit | 1.66 |
| SRAM: working 256 KB, PMEM 32 KB, capture 64 KB, config 32 KB (3.15 Mbit) | 0.2 um2 per bit | 0.63 |
| **Core area** | | **2.73 to 2.76** |
| Pads (14 signal + power), power switches, oscillator, integration | estimate | 0.5 to 0.9 |
| **Die** | | **about 3.2 to 3.7** |

This sits inside PLAN.md's 2.5 to 4 mm² envelope. The ROM is 60% of the core area.

## 6. Listening power at 28 nm (estimate)

Method: Poinsetta's GF180 activity-based measurement of the same front end, VAD and capture logic at the same
3.072 MHz clock (reports/p1-power.md in Poinsetta: 0.69 mW at GF180), scaled to 28 nm with Poinsetta's factor
range (0.0183 to 0.133, from switched capacitance and supply). Marigold's island changes from Poinsetta are added
at GF180-equivalent cost, then scaled the same way. The wake spotter uses the wake agent's own 28 nm estimate.

| Item (listening in silence, core switched off) | GF180-equivalent | 28 nm |
|---|---|---|
| PDM front end (measured in Poinsetta; reused unchanged) | 0.61 mW | |
| VAD, capture logic, ring writes (measured in Poinsetta) | 0.066 mW | |
| Ring leakage, 3.4 of 16 powered banks (pre-roll 4,800 samples; Poinsetta 6.6 banks = 0.014 mW) | 0.007 mW | |
| Config SRAM 32 KB retention (same per-KB leakage as the ring) | 0.017 mW | |
| Island control idle (clock gated; leakage and I2C synchronisers) | about 0.005 mW | |
| Subtotal | 0.705 mW | 12.9 to 94 uW |
| Wake spotter v2, VAD-gated, 20% speech duty (wake agent) | | 7 to 9 uW |
| **Island total** | | **20 to 103 uW** |

The PDM front end is 85% of the island and decides whether target 9 is met; the next lever is its 3.072 MHz CIC
integrators. These new blocks were not run through the GF180 placement and activity flow (Poinsetta's rtl/syn,
rtl/pd), so their figures are activity and leakage estimates, not measurements; that flow is an iteration-2 item.

## 7. Scope cuts and iteration-2 items

- SPI-flash config boot: cut, per the coordinator's scope ruling. spi_flash_reader.sv is in the tree.
- MAC-lane energy redesign: Poinsetta's lanes are reused unchanged, at about 27 pJ per multiply at 45 nm. The
  handoff names this as the lever for transcription-time energy.
- Run the GF180 activity and placement flow on the new island blocks and the wake spotter.
- Memory power-down randomisation for PMEM and working RAM in simulation, as the capture ring already does. The
  firmware reloads PMEM on every boot and does not read RAM before writing it, but the simulation does not prove that.

## 8. How to reproduce (WSL, after `. rtl/core/tools/env.sh`)

- Block tests: `make -C rtl test BLOCKS="pdm vad flash uart mac ram cfg ctl cap"`.
- Core units: `make -C rtl/top test-fan test-mac UIMG=... UTSIMG=...`, both set to the export's data/core/mg-<tag>.
- Image: `make -C rtl/top TAG=<tag> B=$CORE_BUILD/<dir> image-mg`. Install data/core/mg-<tag>/imh_rom.svh as
  rtl/core/imh_rom.svh.
- Chip: `make -f rtl/top/chip.mk sim BSUF=<x> STUB=0 IMG=<flash.hex>`.
- Check: `python3 rtl/top/tb/mg_e2e.py --sim ... --img ... --golden data/golden/mg-<tag>.npz --audio <x.txt>,<x.txt>
  --gaps 0.6,1.5,1.5 --blob <blob with host I2C> --wakes golden --out ... [--neg rom|mic|crc]`.
- Area: `bash rtl/syn/area.sh`.

## 9. v1 rerun

To be filled when the frozen v1 export lands.
