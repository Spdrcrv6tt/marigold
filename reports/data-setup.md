# Data setup (SDK and data agent, 2026-11-15)

What is on disk for Marigold training and evaluation, how much, under which license, and how to rebuild it. Large data lives under data/ (gitignored); the scripts are in sw/data/ (re-included in .gitignore). Licenses in detail: docs/licenses.md.

## What exists

| Set | Where | Count | Audio (measured) | License | Use |
|---|---|---|---|---|---|
| Speech Commands v0.02 | data/speech_commands | 105,829 clips, 35 words, + 6 background-noise files | 28.84 h | CC BY 4.0 | train and eval |
| SLURP real audio | data/slurp/slurp_real | 72,395 FLAC files (34,663 headset, 37,732 far-field) | 57.38 h | CC BY-NC 4.0 | **evaluation only** |
| SLURP text annotations | data/slurp/{train,devel,test,train_synthetic}.jsonl | 11,514 / 2,033 / 2,974 / 19,711 lines | | CC BY 4.0 | labels for the SLURP eval audio |
| MUSAN | data/musan/musan | 2,037 files; 929 noise and 327 music files passed the license check | kept: noise 6.2 h, music 20.5 h | per file | noise pools only |
| TTS eval speech | data/tts/eval/manifest.jsonl | 3,191 clips: 984 eval/text lines x 3 eval voices (2,951: dev 828, test 2,123; 1 rejected) + 80 config examples x 3 eval voices (240, split "typed", target 1) | 1.52 h | ours, from licensed voices | **evaluation only** |
| Noisy eval copies | data/tts/eval_noisy/{fan10db,microwave10db} | 2 x 3,191 clips at 10 dB SNR | 1.97 h each | ours | **evaluation only** (target 10) |
| TTS train speech | data/tts/train/manifest.jsonl | 26,339 clips from 8,780 lines (32 training appliances; no fan or microwave text) | 13.24 h (Kokoro 7.88, Piper 5.36) | ours, from licensed voices | training |
| Noise pools | data/noise/index.jsonl | see Noise | | see Noise | eval and train pools, disjoint |

LibriSpeech stays where it was (data/librispeech, data/librispeech-extra junctions); nothing was copied. The Speech Commands, SLURP and MUSAN tarballs were deleted only after the extracted file count matched the tar listing (105,841, 72,395 and 2,037 files; data/logs/extract_*.count).

## Synthetic speech

**Engines.** Piper 1.8.0 (GPL-3.0 program, used only as a tool) with 9 single-speaker voices and the 904-speaker en_US-libritts-high model, and Kokoro-82M v1.0 (Apache 2.0) with its 28 English voice packs. Each Piper voice's MODEL_CARD was read: a voice is used only if its data and every model it was fine-tuned from allow commercial use. That rules out most Piper multi-speaker voices (libritts_r, vctk, arctic and aru are fine-tuned from the non-commercial lessac voice). Accepted and rejected lists: docs/licenses.md.

**Voice split** (data/tts/voices.json; disjoint by person, asserted in code):
- Eval, 21 speakers: 14 Kokoro (af_heart, af_nicole, af_kore, af_sky, am_michael, am_puck, am_eric, am_onyx, am_fenrir, bf_emma, bf_lily, bm_george, bm_lewis, bm_daniel) and 7 Piper (ljspeech, kristin, john, joe, mike, alba, jenny_dioco); 10 female, 11 male; 14 US English, 7 British English.
- Train, 920 voices from 919 people: 14 Kokoro, the 904 LibriTTS readers, and Cori Samuel (two Piper models). The LibriTTS readers are LibriSpeech train-clean-360 speakers whose real speech already trains the encoder, so none of them is an eval voice. All 920 were used.
- en_US-norman-medium was dropped: license clean, but Whisper WER 50% on the eval phrases (it swallows word onsets).

**Eval set.** Every line of eval/text/paraphrases.txt (700 lines: 180 dev, 520 test) and eval/text/noncommands.txt (284 lines: 96 dev, 188 test), each spoken by 3 eval voices with both engines represented, chosen by a hash of the line. Split, product, action, values and non-command category are in the manifest. Speaking rate varies (length scale 0.9 to 1.15). One clip was rejected as a duration outlier (kristin saying "off" in 2.6 s, garbled tail), so that line has 2 voices.

**Train set.** data/gate0b/train_text.json only (32 training appliances, commands and non-commands; 4,607 clips are non-commands), 3 train voices per line: one Kokoro plus two drawn from all train voices, mostly LibriTTS. Rate 0.85 to 1.25 and randomized Piper noise settings. Per the held-out decision of Nov 15 no fan or microwave text is synthesized for training, not even the config examples: those are spoken by eval voices only (split "typed"). Lines a later data/gate0b/train_text_v3.json adds are picked up by rerunning sw/data/run_train.sh; existing clips are kept.

**Guards.** No eval voice speaks a train line and no train voice speaks an eval line (asserted). Train lines dropped before synthesis: 513 whose normalized text equals an eval/text line or a config example, and the 22 Chinese lines (data/tts/train_dropped_eval_overlap.jsonl). Clips longer than 1.5 s + 0.15 s per spoken character are rejected at merge (1 eval, 1 train: a 51 s LibriTTS clip for a 40-character line; data/tts/*/rejected.jsonl).

**Findings for the model agent (sw/gate0b, not changed by me):**
- data/gate0b/train_text.json contains 332 lines identical to eval/text test lines (108 distinct texts, such as "fire it up", "halt", "begin cooking", "dim the lights") and 102 identical to dev lines, under other products' labels. sw/gate0b does not filter them, so text training on that file sees exact test phrasings.
- It also contains 22 lines in Chinese (for example "不再需要加热了").
- Two config examples equal the other product's test lines: "power off" in fan.json is a microwave stop test line and "stop" in microwave.json is a fan off test line. They are spoken only by eval voices (typed split).

**Format.** 16 kHz mono 16-bit PCM WAV, resampled from 22.05 kHz (Piper) or 24 kHz (Kokoro), peak-normalized to -3 dBFS. Manifests are JSONL: path, text, product, action, values, split, voice, engine, set, category, locale, gender, length_scale, duration_s, plus Whisper QA fields asr_hyp and asr_wer.

**Intelligibility check (QA only).** openai/whisper-base.en (MIT) transcribed every clip. Word error rate with per-clip errors capped at 100% (Whisper sometimes loops on short clips):
- Eval: WER 3.7% over 3,191 clips; per voice 0.75% (Kokoro af_heart) to 6.1% (Piper ljspeech) on the eval/text clips; 112 clips (3.5%) above 50%, mostly one- or two-word commands ("off", "begin") that Whisper mishears without context.
- Train: WER 6.3% over 26,339 clips; Kokoro voices 1.3% to 8.4%, LibriTTS readers 8.5% on average with a tail (the worst six, 24 to 36%, have 7 to 18 clips each); 923 clips (3.5%) above 50%.
Whisper's own mistakes are included, so these are upper bounds on TTS errors, not a TTS measurement. Clips are not dropped on WER; asr_wer lets a consumer do that.

## Noise

| Pool | Eval files | Train files | Source | License |
|---|---|---|---|---|
| fan (**synthetic**) | 6 x 30 s | 24 x 30 s | sw/data/noise_gen.py | ours |
| microwave (**synthetic**) | 6 x 30 s | 24 x 30 s | sw/data/noise_gen.py | ours |
| babble | 6 x 30 s | 24 x 30 s | 4 to 8 LibriSpeech talkers (eval: test-clean speakers; train: train-clean-100) | CC BY 4.0 |
| tv | 6 x 30 s | 24 x 30 s | LibriSpeech speech plus a quieter second talker, band-limited 150 to 6,500 Hz and compressed (eval: dev-clean; train: train-clean-100); speech only, no music bed | CC BY 4.0 |
| background | 169 (1.1 h) | 760 (5.1 h) | MUSAN free-sound (843, public domain) and sound-bible (86: 66 CC BY 3.0, 20 public domain) | as listed, attribution per file in data/noise/musan_clean.json |
| music | 48 (2.8 h) | 279 (17.7 h) | MUSAN music, CC BY or public domain only (rejected: 282 ShareAlike, 32 NoDerivs, 19 unlisted or unclear) | as listed |
| household, stationary | | 6 | Speech Commands _background_noise_ | CC BY 4.0 |

**No commercially clean real fan or microwave recording was found** without an account: MUSAN's noise has no appliance labels, Freesound needs an account, and FSD50K (Zenodo; per-clip licenses, "Mechanical fan" and "Microwave oven" labels) ships as about 24 GB of split zips and was not fetched. The fan and microwave pools are therefore synthetic, labelled so in the index, until John's recordings arrive:
- Fan: pink airflow noise low-passed at 1.2 to 3.5 kHz and chopped at the blade-pass rate; blade-pass tone and harmonics (3 to 7 blades, 800 to 1,700 rpm, slow rpm wander); once-per-revolution imbalance; 120 Hz motor hum.
- Microwave: 60 Hz transformer and magnetron hum, strongest at 120 Hz, harmonics to about 2 kHz, soft-clipped; power cycling below full power (on/off over a 20 to 30 s period); cooling fan (5 to 9 blades, 2,000 to 3,200 rpm); turntable motor whine and slow wobble; three cavity and panel resonances (Q 6 to 15).
- Spectral check of eval_00: the fan peaks at its blade-pass frequency (147 Hz); the microwave at 120 Hz, then 240, 336 (its fan's blade pass) and 360 Hz.

**Mixing** (sw/data/mix.py): SNR = 10 log10(P_speech / P_noise), with P_speech the mean power of active 20 ms frames (within 30 dB of the loudest) and P_noise the power of the noise segment used. The noise runs 0.25 s before and after the speech; a hash of the clip picks the noise file and offset; eval speech gets eval-pool noise only. Every eval clip has a fan copy and a microwave copy at 10 dB. The SNR recorded in the manifests is 10.0 dB by construction (same definition), not an independent check.

## How to regenerate

```
sw/data/extract_tarball.sh <tar.gz> <dir> <logprefix>     # per dataset; deletes the tarball only on a count match
curl -sSfL -o data/slurp/<f>.jsonl https://raw.githubusercontent.com/pswietojanski/slurp/master/dataset/slurp/<f>.jsonl   # f = train, devel, test, train_synthetic
sw/data/tts_fetch.sh                                      # Piper voices and Kokoro-82M weights
.venv/Scripts/python.exe -I sw/data/musan_licenses.py     # MUSAN per-file license filter
.venv/Scripts/python.exe -I sw/data/noise_gen.py synth    # synthetic fan and microwave
.venv/Scripts/python.exe -I sw/data/noise_gen.py speech   # babble and TV
.venv/Scripts/python.exe -I sw/data/noise_gen.py index    # data/noise/index.jsonl
sw/data/run_eval.sh                                       # voices.json, jobs, eval synthesis, manifest, Whisper QA, noisy copies
sw/data/run_train.sh                                      # train jobs, synthesis, manifest, Whisper QA (resumable)
.venv/Scripts/python.exe -I sw/data/mix.py MANIFEST NOISE_TYPE SNR OUT_DIR [--split train]   # any other mix
```
Everything is hash-seeded, so a rerun gives the same voices, assignments and noise. Packages (uv, in .venv): piper-tts, kokoro, en_core_web_sm 3.8.0, soundfile, onnxruntime. Kokoro ran on the GPU (about 1 GB per process), Piper and the noise tools on the CPU. Synthesis took about 2 min for eval and 15 min for train.

## Open

- John's fan and microwave recordings replace the synthetic pools: put them in data/noise/<type>/ and add rows with split eval or train to data/noise/index.jsonl.
- Accents are US and British English only; other accents need real speakers.
- The train_text overlap with eval/text and its Chinese lines (above) need a decision by the model agent or the coordinator.
- Kokoro's model card says its training data includes synthetic audio from closed TTS providers; counsel should confirm before a shipped encoder depends on it.
