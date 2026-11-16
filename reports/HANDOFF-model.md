# Model agent handoff, iteration 2 (Nov 16, 2026, about 12:25 PM)

Stopped early: the run was asked to hand back before any result. Usage at stop: about 2% of the week (meter).
Nothing is frozen yet; no teacher-v4, no model-v2. Detached jobs still running (started by this agent):

- `sw/teacher4/run_a2.sh` (bash, hidden window): waits for the text generation, then
  - teacher selection chain `sw/teacher4/run_sel.sh`: candidates trained WITHOUT the 10 held-out training products
    (same list as v3), scored on them in the 128-d chip space by `sw/teacher4/sel_t4.py`. Order: baseline
    v3cand-small, Qwen3-Embedding-0.6B (full FT, 3000 steps), bge-large-en-v1.5 (full FT), Qwen3-Embedding-4B (LoRA 16,
    1500 steps, checkpointing), Qwen3-Embedding-8B (LoRA, 1200 steps; skipped if data/logs/t4_skip_q8 exists).
    Results: data/logs/t4_sel.log (one line per model and space), reports/gate0/t4_sel_*.json, logs t4_ft_*.log.
  - TTS: jobs2 (eval/text_v2 x 3 EVAL voices -> data/tts/eval_v2; new training products x 3 TRAIN voices ->
    data/tts/train_v4), synthesis, merge. Log data/logs/t4_tts.log ("tts finished" at the end).
- Text: data/gate0b/train_text_v4.json = v3's 72 products + the 48 remaining products of products_v3.json, written by
  local qwen2.5:14b (unloaded afterwards); no fans, blowers or microwaves (asserted). eval/text_v2 arrived 12:06 and
  every loader now drops lines identical to it (sw/textfilter.py).

## Code ready (not yet run end to end)
- sw/teacher4/finetune_t4.py: v4 recipe = gate-0b supervised contrastive per product (tau 0.1) + LibriSpeech spread
  + opposite-action hard negatives (antonym action pairs weighted x3, antonym-swapped minimal pairs) + additive margin
  0.1, loss computed in the chip space through a trained 128-d head saved as marigold_proj.npz (golden-format 5
  unchanged), LoRA for big bases, instruction prompt stored as the model's default prompt.
- sw/teacher4/text_gate.py MODEL --v2: gate 0 in the chip space with the SDK's own blob (sdk/compile.py prototypes,
  integer scores, threshold and margin from eval/text DEV) plus knn_neg; scores eval/text test and eval/text_v2.
- sw/intent/protos.py TEACHER: decision prototypes (8 per training product-action, 16 reject per product).
- sw/intent/train.py --protos FILE: decision loss = softmax over the product's actions (max over prototypes) plus a
  reject class (best reject prototype or a learned threshold), margin 0.05, tau 0.05; LibriSpeech is rejected against
  every product; Speech Commands words get no decision loss; distillation kept (--aux-w).
- sw/intent/evaluate.py: condition `v2` (data/tts/eval_v2), thresholds still from v1 clean dev.
- sw/data/tts_build.py: `jobs2`, `merge NAMES...`. sw/intent/data_index.py reads data/tts/train_v4.
- sdk/compile.py: STEmbedder on GPU when present, bf16 for teachers over 2 GB; default embedder teacher-v4 when it
  exists (model v1 must keep --embedder data/models/teacher-v3).

## Next steps
1. Pick the base from data/logs/t4_sel.log (chip128 acc at 95% rejection on held-out products), retrain it on all
   products (no --holdout) into data/models/teacher-v4, then `python sw/teacher4/text_gate.py data/models/teacher-v4
   --v2 --out reports/gate0/text_gate_teacher_v4.json` once. Gate: 90% unseen and 95% rejection on both products, v2.
2. `data_index.py`, `targets.py data/models/teacher-v4`, `protos.py data/models/teacher-v4`, then train f6 from
   data/intent/f5-final.pt with `--protos data/intent/protos-teacher-v4.npz --targets data/intent/targets-teacher-v4.npz
   --ppb 2 --lr 3e-4 --ctc-w 1.0`; evaluate with `--teacher data/models/teacher-v4 --conds clean,fan10db,microwave10db,v2`.
3. run_final.sh (change its targets path to teacher-v4) -> mg-v2, then `bash rtl/top/regress_mg.sh v2`.
