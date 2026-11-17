# Marigold (Farallon command chip)

Marigold is Farallon's command chip: an always-listening voice-control chip that OEMs drop into products. It hears a wake word, works out what the person wants in their own words (speech to intent, no transcription), and tells the product what to do over I2C, UART or GPIO. It is Poinsettia's smaller sibling: Poinsettia's always-on island plus a much smaller core, a few mm² at 28 nm. PLAN.md is John's plan; docs/decisions.md overrides it where they differ; docs/iteration-1.md defines done (the 12 targets in PLAN.md). docs/poinsetta-handoff.md lists what to reuse from Poinsetta and what it taught us; read it before building anything.

## Rules
- Spend no money. No sign-ups, no API-token services. Local tools only (package managers like uv, conda-forge and apt are fine; open models and datasets downloaded without an account are fine). An AMD account, board purchases and anything with John's name on it are John's to do.
- Stay inside this folder (D:\marigold, junction ~/Marigold). D:\poinsetta and D:\imhotep are read-only references: read them and copy from them, never write to them. Large read-only datasets may be linked from them by junction instead of copied (D: has about 213 GB free; do not duplicate LibriSpeech).
- In-house design: open models (Moonshine, MIT), open datasets, open EDA tools and PDKs are fine. Don't copy third-party chip RTL. Patent guardrail (from Poinsetta's docs/decisions.md): weights one bit per ROM cell, each weight read and multiplied in logic; no multi-bit ROM cells, no ROM transistors shared across bit lines through mask vias, no precompute-and-select ROM schemes, without patent counsel.
- Data licenses: train only on commercially clean data (LibriSpeech and Speech Commands are CC BY 4.0; synthetic audio from TTS voices whose licenses allow it, checked and recorded). SLURP audio (CC BY-NC 4.0) and Fluent Speech Commands (academic only) are for evaluation only. Record every dataset and model license in docs/licenses.md.
- Models: every Marigold agent runs on Opus 5.5 at max effort. A helper agent you spawn gets model "opus" and effort "max", and only when the task truly needs one. Usage is the scarce resource: no narration, read files in parts, wait for background-job notifications instead of polling.
- Usage budget (John, Nov 16 2026): Marigold gets 15 points of the weekly usage in total for this week's work (raised from 10 by John at 10 PM Nov 16), counted from 1% at 11:49 AM Nov 16 (week resets Monday 9 AM CST). Before every major step read ~/.claude/usage-latest.json (`seven_day.used_percentage`). Agents stop at 14% or more: commit, write where you are in reports/HANDOFF-<area>.md, and return. The coordinator keeps the last points for wrap-up and stops everything at 16%. Other projects share the meter, so this is conservative. Spend tokens only where they buy results: launch GPU jobs detached and wait for notifications.
- GPU: the RTX 4090 belongs to Marigold now (Poinsetta is idle). Never kill processes you did not start.
- Large data (datasets, model weights, venvs) goes under data/ or .venv, both gitignored.
- Git is local only. Commit only your own area's paths (never `git add -A`); retry if the index is locked. End commit messages with the attribution lines your session gives.
- Report numbers honestly. Say what was measured and how; label estimates as estimates. A missed target is a finding, not something to hide.

## Working on this PC (learned the hard way on Poinsetta)
- Never trust that a long job started: within 10 minutes of any launch, check its log or GPU use yourself. Any wait loop or gate must time out and report; on Nov 14 a memory gate read an empty value and looped 9 hours with the GPU idle.
- Keep a 30-minute watchdog on long work (John asked for it on Poinsetta): GPU busy when training should run, logs growing, no stuck gates, memory above 8 GB, usage under 88%.
- Claude Code kills background shells under memory pressure. Launch long jobs detached with log files, keep at least 8 GB free on Windows and in WSL, and check memory before big jobs.
- The WSL distro shuts itself down about 30 s after the last wsl.exe session ends, killing jobs detached inside it; killing the Windows-side wsl.exe also kills its job. For long WSL runs keep a wsl.exe session open for the whole run (a background `wsl.exe -d Ubuntu -- sleep infinity`, stopped afterwards) or run the job as the foreground of a background wsl.exe call.
- wsl.exe mangles `$` and quotes passed on its command line: put WSL commands in script files and run the file.
- Kill only your own processes, by PID, never by pattern.

## Layout and owners (same split as Poinsetta)
- sw/: model agent. Intent model, wake-word model, value head, quantization, integer golden model, accuracy. Also docs/golden-format.md and reports/p0-model.md.
- sdk/: SDK agent (config format, phrase expansion, embedding, config blob compiler).
- rtl/: core agent, all of it (front end, wake spotter, core, matcher, I2C/GPIO, top). Verilator + cocotb, synthesis and power flows. Also docs/architecture.md and reports/p1-core.md.
- docs/plan/: John's plan PDF. docs/ip/: patent notes (John's write-up comes first).

## PC notes
- Windows 10, Git Bash + PowerShell, uv installed. WSL distro Ubuntu (`wsl.exe -d Ubuntu`); its clock runs fast and the firewall blocks WSL to Windows.
- WSL toolchain from Imhotep (Verilator, cocotb, Yosys, OpenROAD, OpenSTA, GF180 PDK, clang) is in ~/.imhotep: use read-only. Marigold's WSL build directories go under ~/.marigold/build. The NanGate45 platform (FreePDK45, Apache 2.0) Poinsetta fetched is at ~/.poinsetta/pdk/orfs: read-only reuse.
- Code copied from Poinsetta has hard-coded /mnt/d/poinsetta and D:\poinsetta paths (scripts, Makefiles, harnesses). Retarget them to this repo before relying on them.
