# SDK config compiler (2026-11-15)

sdk/compile.py turns a product config (sdk/config-format.md) into the config blob v0 of docs/config-blob.md, and parses a blob back. Tests: `python -m pytest -q sdk/tests` (10 pass). Blobs for the reference products are in data/sdk/{fan,microwave}.mgc, each with a JSON report next to it.

## What it does

1. Validates the config: SDK rules from config-format.md plus what the chip accepts (docs/architecture.md 8.5, 8.6): ids 1 to 239 (0xF0 and up are event ids), at most 64 action-table entries, 8 slots, 32 GPIO rules, distinct pins 0 to 7, pulse_ms 1 to 65535, latch level 0 or 1, D at most 256.
2. Gathers each action's phrases: the typed examples plus the SDK expansion of that config (data/expand/expanded_qwen25.json from sdk/expand.py; it never sees eval/).
3. Embeds them with the command embedder (data/models/teacher-v3 when it exists, otherwise the newest data/gate0b/models/*; now bge-base-cmd-v2), projects to D = 256 and normalizes.
4. Clusters each action into prototypes (spherical k-means) and the expansion's non-commands into 16 reject prototypes (action 0), filling the 32 KB budget: 18 prototypes per fan action, 27 per microwave action, N = 124.
5. Quantizes vectors to int8, simulates the encoder output in int8, and sets reject_thr so that 90% of the expansion non-commands are rejected under the exact matching rule (0.90 was chosen on the eval dev split; see below). margin = 0 by default.
6. Packs the tables: slots, GPIO rules (pulse level defaults to 1, binary mode carries the slot index), events, wake template, CRC-32. parse() applies the chip's 8.6 checks; pack(parse(blob)) reproduces the blob byte for byte.

Events (config-blob.md, Nov 15): an optional `events` block in the config (`wake`, `nomatch`, `timeout`, each with `host` and optional `gpio`). Host flags go to bits 1 to 3; events with GPIO rules become action-table entries 0xF0 to 0xF2 after the actions. The fan pulses pin 4 for 150 ms on NOMATCH; the microwave sends WAKE, NOMATCH and TIMEOUT to its controller. host_frame() and crc8() build the host frame (CRC-8/SMBus, check value 0xF4 tested).

## Blobs and their text-path accuracy

eval/sdk/blob_text_eval.py embeds the eval/text lines with the blob's own embedder, projection and q scale and runs the exact matching rule: what a perfect audio encoder would reach with these blobs. Measured on 2026-11-15 with bge-base-cmd-v2 and the placeholder projection (reports/sdk-blob-text.json):

| Product | Blob | Paraphrases correct, dev / test | Paraphrases rejected, test | Non-commands rejected (chat+near+tv), dev / test | Other-device commands rejected, test |
|---|---|---|---|---|---|
| fan | 32,383 B, N 124 | 77.8% / 76.6% | 10.3% | 93.9% / 95.7% | 53.8% |
| microwave | 32,328 B, N 124 | 90.3% / 83.2% | 9.1% | 95.1% / 94.4% | 53.8% |

Gate 0's float bound was 88 to 89% unseen at 91 to 93% rejection. The blob path loses accuracy, the fan most. Suspects, not yet separated: the placeholder random projection from 768 to 256 dimensions, prototypes instead of all phrases, int8, and a threshold calibrated on LLM-written non-commands rather than real ones. The reject rate was picked on the dev split only (0.95 rejected 96 to 99% of dev non-commands but cut dev accuracy to 69 to 71%; 0.80 dropped dev rejection to 65 to 79%); test numbers were not used for any choice.

## Placeholders (until docs/golden-format.md and docs/golden-wake.md)

- Projection: fixed seeded random orthonormal 768 to 256 matrix, the same for every product; `--projection FILE.npy` takes the model agent's matrix.
- q scale (the encoder's int8 output scale): 127 / max component of the calibration phrases; `--q-scale` sets it.
- Wake template: byte 0x00 then the ASCII phrase; wake_sens = sensitivity x 65535.
- Biases are 0 and margin 0.

## Next

- When the model agent freezes teacher-v3 (data/models/teacher-v3) and documents the projection and quantization: `python sdk/compile.py sdk/configs/<p>.json --projection <matrix.npy> --q-scale <S>` for both products, then rerun eval/sdk/blob_text_eval.py.
- v1: config fields for the listen window, onset and hangover (architecture 8.1; chip defaults in v0), and the value-head word tables (kept only in the JSON report now).
