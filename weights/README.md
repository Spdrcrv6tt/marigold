# Marigold model weights

These are the weights the chip runs, plus the SDK-side text projections. Larger float files and checkpoints are assets on the GitHub release `model-v2-weights`.

## In this folder

| File | What it is |
|---|---|
| `mg-v3.npz`, `mg-v3.json` | **Intent model v3, as it sits in the chip's weight ROM (shipping since Nov 17, 2026).** Same format, architecture and size as v2 (golden format v1.0, 4,586,560 weights, weight ROM 18,506,240 bits). The same 2-layer student, distilled from a PC-side 6-layer Moonshine audio teacher (reports/p0-model.md section 10). Built from checkpoint qv3.pt (QAT of f7). Tag `model-v3`. Its float twin, vectors and checkpoints are not on a release yet. |
| `mg-v2.npz`, `mg-v2.json` | Intent model v2 (shipped Nov 16 to 17, replaced by v3). Integer golden model, golden format v1.0 (docs/golden-format.md). Moonshine Streaming Tiny frontend plus 2 encoder layers, D = 128 head, 4,586,560 weights (4-bit; the frontend linear and CTC are 8-bit), weight ROM 18,506,240 bits (see the .json). Built from checkpoint qv2.pt (QAT of f6). Tag `model-v2`. |
| `mg-v1.npz`, `mg-v1.json` | Intent model v1 (iteration 1), same format and size. Tag `model-v1`. |
| `teacher-v4_proj.npz` | The 128-d chip-space projection on top of teacher-v4 (Qwen3-Embedding-4B + LoRA 16). Used only by the SDK on a PC, to embed OEM phrases. Pairs with mg-v2 and mg-v3. |
| `teacher-v3_proj.npz` | Same for teacher-v3, which pairs with mg-v1. |
| `SHA256SUMS` | Checksums (of the files as stored in git; the .json files are LF since Nov 17). |

The `.json` next to each `.npz` gives the config, per-tensor bit widths, the weight count and the ROM size.

## Already elsewhere in the repo

- **Wake spotter v2 (format 2, 319k weights):** `sw/wake/golden/wake_int_v2.npz`. It is identical word for word to `rtl/wake/mem2/wake_rom.mem`. Format 1 is `sw/wake/golden/wake_int.npz`. Rules are in docs/golden-wake.md.
- **Integer reference implementations:** `sw/golden/` (intent) and `sw/wake/golden/` (wake). The RTL in `rtl/` matches them bit for bit (`rtl/top/regress_mg.sh v2`).

## On the release `model-v2-weights`

| Asset | What it is |
|---|---|
| `mg-v2-float.npz`, `mg-v1-float.npz` | Float twins of the golden models, the same tensors before integer rounding. |
| `vectors-mg-v2-0..3.npz` | Test vectors for mg-v2 (inputs, every intermediate tensor, outputs), used by the RTL unit tests. |
| `f6-final.pt` | Float PyTorch checkpoint of the v2 encoder before QAT. |
| `qv2.pt`, `qv1.pt` | QAT checkpoints exported to mg-v2 and mg-v1. |
| `teacher-v3.tar` | The complete teacher-v3 sentence-transformers folder (SDK side). |

**Not uploaded:** the full teacher-v4. It is 8 GB, Qwen3-Embedding-4B with a LoRA merged in, and it is reproducible with `sw/teacher4/` from the public base model. It can be added on request.

## Licenses and attribution

- **Moonshine Streaming Tiny** (Useful Sensors), MIT. The intent model's frontend and encoder layers were initialised from it and fine-tuned.
- **Qwen3-Embedding-4B** (Alibaba Qwen), Apache 2.0. It is the base of teacher-v4.
- **Training audio:**
  - LibriSpeech (CC BY 4.0, Panayotov et al., openslr.org/12).
  - Speech Commands (CC BY 4.0, Warden).
  - Synthetic speech from commercially licensed Kokoro and Piper voices, listed with their licenses in docs/licenses.md.
- **SLURP** (CC BY-NC) was used only for evaluation, never for training.

No license is granted yet for the Marigold weights or code themselves (all rights reserved).
