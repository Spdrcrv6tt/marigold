# What Marigold takes from Poinsetta

Poinsetta, Farallon's streaming speech-to-text chip, lives at D:\poinsetta (read-only for Marigold). Its final commit is 1719528, and its hardware is unchanged since b2fe8ba. Iteration 1 closed on Nov 15, 2026 with all nine items met: bit-exact whole-chip RTL, an independent rebuild, and 5.05% word error. Start with D:\poinsetta\reports\validation-iteration1.md and D:\poinsetta\docs\spec-sheet.md; every decision and its reason is in D:\poinsetta\docs\decisions.md.

## Reuse map

| Poinsetta path | What it is | For Marigold |
|---|---|---|
| rtl/src/pdm_frontend.sv, cic_decim.sv, fir_decim2.sv, dc_block.sv, pdm_clkgen.sv, pdm_coefs.svh, pdm_*_coef.mem | PDM front end: 3.072 MHz one-bit stream in, 16 kHz samples out | Reuse as is, in the always-on island |
| rtl/src/vad.sv | Voice detector; noise statistics learn only from quiet frames (K_G = 16, K_CAP = 48, golden format 3.5) | Reuse; it gates the wake spotter |
| rtl/top/imh_capture.sv | Capture ring in 32 banks of 4 KB with bank power enables, plus segmentation on the mic clock | Reuse, shrink to 32 to 64 KB; segments start at a wake instead of a VAD onset |
| rtl/core/imh_cpu.sv | RV32IM controller | Reuse |
| rtl/core/imh_fan.sv, rtl/src/mac_lane.sv | Fan-out MAC array (32 lanes x 8 frames, per-frame clock gates) | Reuse with fewer lanes, and redesign for energy (see the numbers below) |
| rtl/core/imh_mac.sv | Attention MAC array (16 lanes) | Reuse if the intent model keeps attention |
| rtl/core/imh_vec.sv, rtl/core/imh_rom.svh | Per-element unit: norms, softmax, lookup tables (asinh, GELU, SiLU, tanh, exp2), requantization | Reuse; the LUT ROM belongs to the ROM image |
| rtl/core/imh_mem.sv | Weight-source ROM port: stall-tolerant, output register clocked per read | Reuse |
| rtl/src/clk_gate.sv, latch_ram.sv, uart_tx.sv, spi_flash_reader.sv | Clock gate, RAM model, UART, SPI flash reader | Reuse; SPI flash reader fits the standalone config load |
| rtl/core/fw/ | Firmware (stream.c runs the per-chunk schedule) | Pattern for the intent pipeline firmware |
| rtl/core/tools/mkimage_ts.py | ROM image builder (4-bit packing, layout.h) | Adapt |
| rtl/top/ts_e2e.sh, regress_ts.sh, session_ts.sh, rtl/top/tb/ | Whole-chip harness, regression with negative controls, continuous session | Adapt: mic pin in, I2C, UART and GPIO out |
| rtl/syn/ (p1_power.sh, activity.py, aon_power.py) | GF180 power flow driven by simulated switching activity | Reuse, keeping activity.py's fix_wide |
| rtl/pd/ng45.py, rtl/pd/pd.sh | NanGate45 per-block flow (45 nm grounding for 28 nm), GF180 layout flow | Reuse for phase 4 |
| sw/golden/ (stream_np.py, stream_export.py, stream_test.py, stream_vectors.py) | Integer golden model, export, exactness tests, test vectors | Pattern for the intent golden model |
| sw/qat.py, sw/quant.py, sw/sensitivity.py | QAT with distillation from a float teacher, noise augmentation | Reuse for the intent encoder |
| sw/latency/ | Chip timing model calibrated on RTL runs | Pattern for target 7 |
| validation/rebuild.sh, validation/word_latency.py | Independent rebuild from a fresh clone; latency from alignments and UART timing | Reuse at the end |
| fpga/fit/ | Yosys fit check for KV260 and others | Reuse once the RTL exists |
| tools/ (block_diagram.py, render_die.py, render_floorplan28.py) | Diagrams, die renders, 28 nm floorplan | Reuse |
| docs/golden-format.md, docs/architecture.md | How Poinsetta wrote its exact integer rules and its hardware design | Templates |

Data already on disk (link by junction, don't copy; D: has about 213 GB free):
- LibriSpeech: D:\imhotep\data\librispeech (linked from D:\poinsetta\data\librispeech), plus train-other-500 at D:\poinsetta\data\librispeech-extra.
- 8-talker babble test set: D:\poinsetta\data\test-clean-babble10.
- Moonshine weights in the Hugging Face cache: D:\poinsetta\data\hf.
- Poinsetta's Python environment: D:\poinsetta\.venv (requirements in sw/requirements.lock.txt). Build Marigold its own .venv with uv.

## Numbers from Poinsetta that shape Marigold

Measured unless marked as an estimate.

- **Always-on island.** 0.69 mW at GF180, 89% of it the PDM front end; 28 nm estimate 13 to 94 µW with core, RAM and ROM switched off. Marigold's target 9 (100 µW) adds a wake spotter to this island, so budget it from day one.
- **28 nm densities** (estimates): ROM 0.06 µm² per bit at 70% array efficiency; SRAM 0.12 µm² per bit at 60%; logic NAND2 0.50 µm² at 70% placement.
- **45 nm grounding** (NanGate45): Poinsetta's blocks reach 200 to 240 MHz at the typical corner. Scaling 45 to 28 nm: delay 0.47 to 0.79x, switching energy 0.42 to 0.53x, area 0.63x.
- **MAC energy is the lever.** Poinsetta's fan-out array spends about 27 pJ per multiply at 45 nm, 89% of the logic's switching power: well above a lean design. Design Marigold's lanes for energy from the start (operand isolation, fewer re-registered operands, narrower accumulators) and measure early.
- **Clock gating** per unit and per frame cut transcribing energy 43%. Build it in from the start.
- **Moonshine Tiny Streaming**: 44M weights. The encoder is 9.46M (stem plus 6 layers, hidden 320, local attention), 50 frames per second, and a frame leaves it 320 ms after its audio. That look-ahead alone exceeds target 7's 300 ms median if kept, so the shrunk encoder needs less look-ahead or a different end-of-speech rule.
- **Noise robustness recipe.** Fine-tuning the 4-bit model with noisy audio while a full-precision teacher hears the clean audio cut 8-talker babble error from 47% to 17% with clean speech slightly better. The same trick fits target 10 (product noise).

## Lessons

1. **Order of work.** Golden model first, with a frozen, written format; then bit-exact RTL. Prove it with a regression plus negative controls (a flipped weight bit, corrupted mic data), a continuous session, and an independent rebuild from a fresh clone.
2. **Power flows can lie.** Poinsetta inherited a flow that ignored register and clock activity (Imhotep's 49 µW figure was wrong), and Verilator 5.054 over-counts toggles on signals wider than 64 bits (rtl/syn/activity.py, fix_wide). Sanity-check any power number against a hand estimate.
3. **Count the ROM image as built.** Tables stored one byte per 4-bit value cost Poinsetta 47 Mbit until they were packed.
4. **Give every memory port a stall from day one.** Poinsetta's working-memory port has no wait signal, so its FPGA demo needs a 6 s cap or an RTL change; the weight port's stall made board DDR4 easy.
5. **FPGA.** The AMD Kria KV260 is the board (D:\poinsetta\reports\fpga-fit.md). John buys it and sets up the AMD account for the free Vivado edition; check Marigold's fit with the same Yosys method once RTL exists.
6. **Only 28 nm matters to John.** GF180 is just the measured basis; its routing of 28 nm-class logic never closed, and the full route was stopped.
7. **Latency comes from the schedule.** Format 3.6's quick tail updates cut Poinsetta's median word-to-text latency from 0.95 to 0.47 s with no hardware change.
8. **Agents and jobs.** See CLAUDE.md: verify every long job within 10 minutes, a 30-minute watchdog, WSL keepalive, memory checks, kill by PID only.

## People and process

- John reaches the PC from his phone. He wants terse updates in plain words, and work that is "done and usable".
- The Poinsetta coordinator session ("SECRET // POINSETTIA" in ListAgents) can answer questions about Poinsetta while it is running.
