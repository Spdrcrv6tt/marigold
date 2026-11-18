# Start here: Marigold coordinator session

You coordinate Marigold, the command chip. John had this session started on Nov 15, 2026 from the Poinsetta coordinator session ("SECRET // POINSETTIA"). In his words: "Poinsettia seems decently built, so I want to move on to this idea. ... I want to reuse as much as we possibly can and use as much as we've learned to build this chip as efficiently as possible. Let's go ahead and get working."

## Read first, in order

1. D:\marigold\CLAUDE.md: the rules. They bind you and every agent you start.
2. D:\marigold\PLAN.md: John's plan. The original is the PDF in docs/plan/.
3. D:\marigold\docs\poinsetta-handoff.md: what to reuse from Poinsetta, the numbers it measured, and its lessons.
4. D:\marigold\docs\decisions.md.
5. Your memory index (MEMORY.md), especially the entries on chip agents (Opus 5.5 max only), verifying that long jobs really run, the usage meter, and [[project-marigold]].

## Your job now: the plan's first session

Work through the "First session" checklist at the end of PLAN.md. Notes:

1. **RTL copy.** The repo exists (D:\marigold, git initialized, junction ~/Marigold). Copy the RTL the handoff lists from Poinsetta commit 1719528 (`git -C D:\poinsetta archive 1719528 <paths>` or `git show`), retarget hard-coded paths, and get the copied unit tests passing in WSL. A core agent can do this.
2. **docs/iteration-1.md** from the 12 targets in PLAN.md.
3. **Decision 1** in docs/decisions.md: speech-to-intent with text-enrolled matching, no transcription.
4. **Config format v0** and the two product configs: the fan and the microwave.
5. **Paraphrase test set:** 50 or more ways to say each action, none copied from the configs' example phrases. Also write ordinary non-command talk for the rejection test.
6. **Text-only upper bound, the first gate.** Embed the examples and test phrases with a small, commercially licensed text-embedding model, match by similarity, and measure accuracy on unseen phrasings and rejection of non-commands. Can text matching alone reach 90% unseen and 95% rejection? Tell John the answer as soon as you have it: it is the go/no-go for the whole approach.
7. **Datasets.** LibriSpeech is already on disk: link it, don't copy it. Download Speech Commands (CC BY 4.0) for training, and SLURP audio for evaluation only. Record every license in docs/licenses.md.
8. **Synthetic command audio.** Use local TTS with many voices whose licenses allow commercial use (check and record each one), mixed with fan, microwave and TV noise from CC0 sources until John's own recordings arrive. Recording 30 minutes each of a real fan and microwave is John's item: ask him once, plainly, for phone recordings when he can.

Work through agents as Poinsetta did: a model agent (sw/), an SDK and data agent (sdk/, data), and a core agent (rtl/). You coordinate: give each a clear brief, record decisions in docs/decisions.md, and keep John posted in short, plain lines.

## Usage and pace

- Weekly usage was 80% at 4 PM CST on Nov 15. The stop is 88%, and the reset is Monday, Nov 16, 9 AM CST.
- Do the cheap first-session items now: the text upper bound, configs, paraphrases, the RTL copy and the downloads. Start GPU training only if usage allows; otherwise after the reset.
- Whenever a long job runs, check it within 10 minutes and keep a 30-minute watchdog (CronCreate) on it. Delete the watchdog when nothing is running.

## When the checklist is done (or usage stops you)

Send John a short summary:
- the text-only result, the go/no-go;
- what is set up;
- what you need from him.

Commit everything. The Poinsetta coordinator ("SECRET // POINSETTIA") can answer questions about Poinsetta while it is running.
