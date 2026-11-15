# Licenses of data and models

Every dataset, model and voice Marigold touches, with its license and what we may use it for. Training uses only commercially clean entries. Checked from the source's own license file or model card on the date shown.

## Datasets

| Name | Source | License | Use | On disk | Checked |
|---|---|---|---|---|---|
| LibriSpeech (all splits, incl. train-other-500) | openslr.org/12 | CC BY 4.0 | Training and eval | data/librispeech, data/librispeech-extra (junctions to Imhotep and Poinsetta copies) | 2026-11-15 |
| Speech Commands v0.02 | download.tensorflow.org/data/speech_commands_v0.02.tar.gz (Warden 2018) | CC BY 4.0 | Training and eval | data/speech_commands | 2026-11-15 |
| SLURP audio (slurp_real) | zenodo.org/records/4274930 | CC BY-NC 4.0 | **Evaluation only** | data/slurp | 2026-11-15 |
| SLURP text annotations | github.com/pswietojanski/slurp | CC BY 4.0 | Evaluation labels | not yet downloaded | 2026-11-15 |
| Fluent Speech Commands | Fluent.ai | Academic research only | **Evaluation only**, if obtained | not downloaded | 2026-11-15 |
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

## TTS voices (synthetic training audio)

None generated yet. Each voice gets a row here, with its own license, before any audio from it is used.
