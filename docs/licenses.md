# Licenses of data and models

Every dataset, model and voice Marigold touches, with its license and what we may use it for. Training uses only commercially clean entries. Checked from the source's own license file or model card on the date shown.

## Datasets

| Name | Source | License | Use | On disk | Checked |
|---|---|---|---|---|---|
| LibriSpeech (all splits, incl. train-other-500) | openslr.org/12 | CC BY 4.0 | Training and eval | data/librispeech, data/librispeech-extra (junctions to Imhotep and Poinsetta copies) | 2026-11-15 |
| Speech Commands v0.02 | download.tensorflow.org/data/speech_commands_v0.02.tar.gz (Warden 2018) | CC BY 4.0 | Training and eval | data/speech_commands | 2026-11-15 |
| SLURP audio (slurp_real) | zenodo.org/records/4274930 | CC BY-NC 4.0 | **Evaluation only** | data/slurp | 2026-11-15 |
| SLURP text annotations | github.com/pswietojanski/slurp (dataset/slurp/*.jsonl; README: text CC BY 4.0, audio CC BY-NC 4.0) | CC BY 4.0 | Evaluation labels (they describe eval-only audio) | data/slurp/{train,devel,test,train_synthetic}.jsonl | 2026-11-15 |
| Fluent Speech Commands | Fluent.ai | Academic research only | **Evaluation only**, if obtained | not downloaded | 2026-11-15 |
| MUSAN | openslr.org/17 (Snyder et al. 2015) | Per file, from MUSAN's own LICENSE files; see Noise below | Only files whose license passed the check | data/musan (musan.tar.gz deleted after a file-count check) | 2026-11-15 |
| CMUdict (cmusphinx/cmudict, cmudict.dict, 135,166 entries) | github.com/cmusphinx/cmudict (raw download, LICENSE file) | BSD-2-Clause style (Copyright 1993-2015 Carnegie Mellon University; redistribution with the notice; commercial use allowed) | Wake: phone labels for training transcripts and the SDK's G2P for typed wake phrases (sw/wake/lexicon.py) | data/wake/lex/cmudict.dict, LICENSE | 2026-11-15 |
| MUSAN speech (librivox, us-gov) and all non-training MUSAN music | openslr.org/17 | Per file (public domain speech; music mixed licenses) | **Evaluation only** (wake false-wake streams; never trained on) | data/musan | 2026-11-15 |
| Marigold paraphrase and non-command sets | eval/text/ (written in house) | Ours | Evaluation only (frozen test sets) | eval/text | 2026-11-15 |

## Models

| Name | Role | License | Notes | Checked |
|---|---|---|---|---|
| sentence-transformers/all-MiniLM-L6-v2 | Gate 0 text embedding candidate | Apache 2.0 | Trained on public pairs incl. MS MARCO (non-commercial terms); weights are Apache 2.0. Counsel should confirm before the SDK ships it. | 2026-11-15 |
| BAAI/bge-small-en-v1.5, bge-base-en-v1.5 | Gate 0 candidates | MIT | Same training-data caveat as above | 2026-11-15 |
| intfloat/e5-small-v2 | Gate 0 candidate | MIT | Same caveat | 2026-11-15 |
| thenlper/gte-small | Gate 0 candidate | MIT | Same caveat | 2026-11-15 |
| Snowflake/snowflake-arctic-embed-s | Gate 0 candidate | Apache 2.0 | Same caveat | 2026-11-15 |
| Qwen2.5-32B-Instruct (Ollama qwen2.5:32b-instruct-q4_K_M) | SDK phrase expansion on the PC | Apache 2.0 | Runs locally; generates SDK-side paraphrases only | 2026-11-15 |
| Moonshine (Useful Sensors) | Encoder starting point | MIT | Weights in D:\poinsetta\data\hf | 2026-11-15 |

## TTS engines and tools (run locally as tools; nothing here is linked into Marigold or shipped)

| Name | Role | License | Notes | Checked |
|---|---|---|---|---|
| piper-tts 1.8.0 (OHF-Voice/piper1-gpl), bundles espeak-ng | TTS engine for Piper voices | GPL-3.0-or-later (espeak-ng GPL-3.0) | Used as a program to make audio; the audio is not a derivative of the program. Never linked into Marigold code. | 2026-11-15 |
| kokoro 0.9.4 + misaki 0.9.4 (G2P) | TTS engine for Kokoro voices | Apache 2.0 | misaki falls back to espeak-ng (phonemizer-fork 3.3.2, GPL-3.0+, espeakng-loader) for unknown words; tool use only | 2026-11-15 |
| spaCy 3.8 + en_core_web_sm 3.8.0 | misaki tokenizer/tagger | MIT | | 2026-11-15 |
| onnxruntime 1.32, transformers 5.23, torch 2.6 | runtimes | MIT, Apache 2.0, BSD-3 | | 2026-11-15 |
| openai/whisper-base.en | QA only: intelligibility check of synthetic speech (sw/data/tts_asr_check.py) | MIT | Its transcripts are never training data | 2026-11-15 |

## TTS voices (synthetic training audio)

Every voice was checked against its own model card (data/tts/piper/cards/*.md for Piper; the Kokoro-82M model card). A voice is accepted only when its training data allows commercial use **and** every model it was fine-tuned from does too. Piper voice weights: MIT (rhasspy/piper-voices repository license). Split: data/tts/voices.json, written by sw/data/tts_build.py.

### Accepted

| Voice | Speakers | Training data and license | Lineage | Split |
|---|---|---|---|---|
| Kokoro-82M v1.0 (hexgrad), 28 English voice packs (af_*, am_*, bf_*, bm_*) | 28 | Weights Apache 2.0. Card: trained only on permissive or non-copyrighted audio (public domain, Apache/MIT-licensed audio, synthetic audio from closed TTS providers; none from open models or voice clones). Counsel should confirm the synthetic-audio provenance before a shipped encoder depends on it (same caveat as the Gate 0 embedders). | own training | 14 eval, 14 train |
| Piper en_US-libritts-high | 904 | LibriTTS train-clean-360 (openslr.org/60), CC BY 4.0 | trained from scratch | train only: its readers are LibriSpeech train-clean-360 speakers, whose real speech trains the encoder |
| Piper en_US-ljspeech-high | 1 (F) | LJ Speech, public domain | from scratch | eval |
| Piper en_US-kristin-medium | 1 (F) | LibriVox recordings, public domain | from scratch | eval |
| Piper en_US-john-medium | 1 (M) | LibriVox recordings, public domain | fine-tuned from kristin-medium | eval |
| Piper en_US-joe-medium | 1 (M by F0) | OHF-Voice/voice-datasets, CC0 | fine-tuned from rdh base (nl_BE, CC0) | eval |
| Piper en_US-mike-medium | 1 (M by F0) | OHF-Voice/voice-datasets, CC0 | fine-tuned from rdh base (CC0) | eval |
| Piper en_GB-alba-medium | 1 (F by F0) | Edinburgh DataShare 10283/3270, CC BY 4.0 | fine-tuned from hifi_tts_92 base | eval |
| Piper en_GB-jenny_dioco-medium | 1 (F) | Jenny (Dioco) dataset license: commercial use permitted; products that generate audio with the voice on user request must credit "Jenny (Dioco)"; no attribution needed for distributing generated clips. We only make clips. | fine-tuned from hifi_tts_92 base | eval |
| Piper en_GB-cori-high | 1 (F, Cori Samuel) | LibriVox recordings, public domain | from scratch | train |
| Piper en_US-hifi_tts_92-medium | same person as cori-high | Hi-Fi TTS reader 92 (NVIDIA, openslr.org/109), CC BY 4.0 | from scratch | train (kept with cori-high: the split is by person) |

### Rejected

| Voice(s) | Reason |
|---|---|
| en_US-lessac (all qualities) | Blizzard 2013 Lessac license, non-commercial |
| en_US-libritts_r-medium, en_GB-vctk-medium, en_US-arctic-medium, en_GB-aru-medium, en_GB-northern_english_male, en_US-sam, en_US-reza_ibrahim, en_US-kusal, en_GB-alan | Fine-tuned from lessac, so non-commercial lineage, even where their own data is CC BY, CC0 or Apache |
| en_US-l2arctic | L2-ARCTIC, CC BY-NC 4.0 |
| en_US-hfc_female, en_US-hfc_male | Hi-Fi-CAPTAIN, CC BY-NC-SA 4.0 |
| en_US-ryan (all) | RyanSpeech, CC BY-NC-SA 4.0 |
| en_US-kathleen-low, en_US-danny-low, en_GB-southern_english_female-low | Fine-tuned from ryan-low (non-commercial lineage) |
| en_US-amy | Dataset license "see URL" (mimic3-voices), unclear |
| en_US-bryce | Fine-tuned from an unreleased voice of unknown lineage |
| en_GB-semaine | SEMAINE database (academic licence); not used |
| en_US-norman-medium | License clean (LibriVox, public domain, from scratch) but dropped for quality: Whisper WER 50% on the eval phrases with any noise setting (it swallows word onsets) |

### Voice split (data/tts/voices.json)

- **Eval, 21 speakers** (held out; the only voices that speak eval/text lines): Kokoro af_heart, af_nicole, af_kore, af_sky, am_michael, am_puck, am_eric, am_onyx, am_fenrir, bf_emma, bf_lily, bm_george, bm_lewis, bm_daniel; Piper ljspeech, kristin, john, joe, mike, alba, jenny_dioco. 2 engines (14 Kokoro, 7 Piper); 10 female, 11 male; 14 en-US, 7 en-GB. Median F0 per voice is in voices.json.
- **Train, 920 voices / 919 people**: Kokoro af_bella, af_aoede, af_sarah, af_alloy, af_nova, af_jessica, af_river, am_echo, am_liam, am_santa, am_adam, bf_isabella, bf_alice, bm_fable; Piper libritts-high (904 readers), cori-high and hifi_tts_92 (Cori Samuel).
- Eval voices also speak the fan and microwave config examples (eval split "typed", target 1); train audio never contains fan or microwave text (decision of Nov 15).
- The sets are disjoint by person (checked in code). Accent coverage is only US and British English; Indian, Australian, Southern US, Spanish-accented and other speakers need real recordings.


## Noise (data/noise/index.jsonl; built by sw/data/noise_gen.py and sw/data/musan_licenses.py)

| Source | Type | License | Use | Checked |
|---|---|---|---|---|
| Procedural fan and microwave noise (sw/data/noise_gen.py) | fan, microwave | Ours (generated) | **Synthetic** stand-in for target 10 until John's recordings arrive; separate eval (6 each) and train (24 each) seeds | 2026-11-15 |
| Babble and TV-like noise built from LibriSpeech | babble, tv | CC BY 4.0 (LibriSpeech) | Eval pool from test-clean (babble) and dev-clean (TV) speakers, train pool from train-clean-100 | 2026-11-15 |
| MUSAN noise/free-sound (843 files) | background | Public domain (MUSAN LICENSE: all marked public domain on Freesound) | Train/eval pools (1 in 5 to eval by file id) | 2026-11-15 |
| MUSAN noise/sound-bible: 86 kept (66 CC BY 3.0, 20 public domain) | background | CC BY 3.0 / public domain; attribution kept per file in data/noise/musan_clean.json | Train/eval pools | 2026-11-15 |
| MUSAN music: 327 kept (CC BY: rfm 147 Kevin MacLeod, fma 78, fma-western-art 29, jamendo 18, hd-classical 1; public domain: fma-western-art 29, hd-classical 18, fma 7) | music | CC BY 3.0/4.0 or public domain; attribution per file in musan_clean.json | Train/eval pools | 2026-11-15 |
| MUSAN rejected: 333 music files + 1 sound-bible file | | ShareAlike 282, NoDerivs 32, no license found 12, not listed in LICENSE 7 music + 1 sound-bible (data/noise/musan_rejected.json) | Not used | 2026-11-15 |
| Speech Commands _background_noise_ (6 files) | household, stationary | CC BY 4.0 | Train pool | 2026-11-15 |

No commercially clean real fan or microwave recording was found without an account: MUSAN's noise annotations carry no appliance labels, and Freesound needs an account. FSD50K (Zenodo) has per-clip licenses and "Mechanical fan" / "Microwave oven" labels but ships as ~24 GB of split zips; not fetched.
