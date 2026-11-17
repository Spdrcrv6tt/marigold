# Gate 0: text-only upper bound (Nov 15, 2026)

**Question:** can text matching alone reach 90% on phrasings the OEM never typed (target 2) and reject 95% of non-commands (target 4)?

**Answer: not yet.** No off-the-shelf text-embedding model reaches both. The best result, a 335M-weight embedder with SDK phrase expansion and a linear matcher, gets 88 to 89% unseen accuracy at 91 to 93% rejection. Small embedders (22M to 33M) get 83 to 88% at about 90%. The misses come from the embedding space, not from audio, so the fix belongs in the SDK's text model (next step below) before any audio training.

## Setup

- Configs: sdk/configs/fan.json (6 actions) and microwave.json (4 actions), 8 typed examples per action.
- Test sets (frozen and committed in 79e4114 before any result): eval/text/paraphrases.txt, 70 per action, 52 per action in test (520 total); eval/text/noncommands.txt, 244 lines in chat, near-miss and TV categories (162 in test), plus 40 commands meant for other devices (26 in test), reported separately.
- Threshold rule, fixed in advance: per product, the lowest threshold that rejects 95% of the dev non-commands; test numbers use it unchanged. A rejected command counts as wrong.
- SDK expansion: local Qwen2.5-32B-Instruct (Apache 2.0, via Ollama) wrote about 160 paraphrases per action and about 190 non-command lines per product from the config alone (sdk/expand.py). It never saw eval/. 41 of the 520 test phrases also came out of the expansion; they stay in the test set and are counted here.
- Matchers: `max` = nearest example; `knn_neg` = nearest neighbour, rejected if it is a generated non-command; `linear` = one weight vector per action plus a "none" class (logistic regression), which fits the 32 KB config SRAM; `centroid` = one mean vector per action.
- Script: eval/gate0/text_bound.py. Raw results: reports/gate0/*.json.

## Results (test split; unseen accuracy / headline rejection)

| Embedder (license, size) | Typed examples only, `max` | + expansion, `max` | + expansion, `knn_neg` | + expansion, `linear` |
|---|---|---|---|---|
| all-MiniLM-L6-v2 (Apache, 22M) | fan 50% / 96%, mw 59% / 94% | 67 / 95, 74 / 95 | 69 / 94, 78 / 94 | 82 / 88, 85 / 91 |
| bge-small-en-v1.5 (MIT, 33M) | 64 / 94, 81 / 96 | 77 / 94, 80 / 94 | 77 / 94, 79 / 97 | 83 / 91, 88 / 91 |
| e5-small-v2 (MIT, 33M) | 65 / 95, 76 / 96 | 80 / 93, 82 / 96 | 81 / 91, 81 / 96 | 83 / 90, 88 / 90 |
| gte-small (MIT, 33M) | 59 / 94, 76 / 92 | 76 / 91, 81 / 96 | 76 / 92, 83 / 94 | 81 / 90, 82 / 93 |
| arctic-embed-s (Apache, 33M) | 42 / 97, 60 / 98 | 57 / 98, 59 / 97 | 57 / 98, 61 / 97 | 85 / 83, 85 / 91 |
| bge-base-en-v1.5 (MIT, 110M) | 73 / 91, 81 / 93 | 79 / 88, 82 / 96 | 79 / 90, 83 / 93 | 87 / 85, 85 / 95 |
| bge-large-en-v1.5 (MIT, 335M) | | | 81 / 94, 85 / 96 | **88 / 91, 89 / 93** |
| mxbai-embed-large-v1 (Apache, 335M) | | | 76 / 96, 82 / 98 | 87 / 93, 89 / 91 |

With rejection switched off, the best accuracy is 89 to 94%, so even a perfect reject path would leave the fan short of 90% with these embedders.

## Where it fails (e5-small, `linear`, fan)

- **Indirect phrasings:** "i'm roasting", "it's stuffy in here", "i've got goosebumps", "my papers are blowing away" are rejected, because the generated non-commands hold similar complaints.
- **Opposites sit close together:** "too weak" goes to slower; "stop the rotation" and "turn off the rotation" go to off; "power down the fan" goes to slower.
- **Single words and level names:** "high", "low", "lowest setting" and "full speed" go to faster, slower or reject instead of set_level.
- **False accepts:** "level three of the game is hard" goes to set_level, "never mind" to off, "the oscillating sprinkler is broken" to oscillate.
- Some test labels are genuinely ambiguous: "power up" could be on or faster, and "low" could be set_level or slower. They were labelled before any result and stay as labelled.

## What it means and what's next

The plan's idea holds up in shape: open phrasing works for most direct commands. But general-purpose sentence embedders aren't built to separate "faster" from "slower" or a complaint from small talk. The text model lives only in the SDK, and the chip's audio encoder will be trained to land on its vectors, so it can be any model we like.

- **Gate 0b, the next experiment:** fine-tune a small embedder (bge-small or e5-small) for commands, with contrastive pairs and opposite-action hard negatives. Train it on LLM-generated command and non-command data for 30 or more other products (purifier, heater, lamp, oven, washer and so on). Fans and microwaves stay out of its training data entirely, which also tests target 11 honestly.
- The fine-tuned space is also an easier target for a 2M to 5M weight audio encoder.
- Fallbacks if 0b also falls short: a narrower phrasing promise (direct commands; indirect complaints as an opt-in with OEM examples), or more config SRAM for more prototypes per action.

## Gate 0b, first runs (same evening)

The SDK's text embedder was fine-tuned for commands (sw/gate0b/finetune_text.py) on text a local LLM wrote for 32 other appliances (sw/gate0b/products.json; 7,834 command phrasings, 1,542 non-commands; no fans or microwaves, so both stay new products). The recipe is a supervised contrastive loss within one product per batch. Diagnosis used the dev split only.

| Run | Fan, unseen / rejection | Microwave, unseen / rejection |
|---|---|---|
| bge-small, untuned, + expansion, `knn_neg` (from above) | 77% / 94% | 79% / 97% |
| bge-small tuned v1 (tau 0.05) | 81 / 94 | 87 / 86 |
| bge-small tuned v2 (tau 0.1, plus LibriSpeech snippets as their own items so ordinary speech stays spread out) | 83 / 94 | 92 / 90 |
| bge-base tuned v2, `knn_neg` | **83 / 96** | **90 / 93** |

v1 collapsed the space (dev: "okay", "hmm" and "nothing" scored 0.95 or more against a command); v2 fixed most of that.

- **Microwave** is now at the bar for unseen phrasings and 2 points short on rejection.
- **Fan** rejection passes; unseen accuracy is 7 points short, mostly from oscillate. "Stop rotating" goes to off and "spin around" to on: none of the 32 training products has a swing or rotate mode.
- Without rejection, accuracy is 89% (fan) and 95% (microwave).
- Still unmet as a pair, but tuning the SDK-side model moved it 5 to 10 points in an hour. The approach looks fixable; it is not dead.

Next levers, for the model agent:
- Many more training products and action types (100 or more, including motion and toggle modes such as AC swing).
- More non-command data.
- Model choice on held-out training products instead of eval dev.
- A margin rule (best minus second-best) next to the threshold.
- Combine the kNN and linear matchers.

## Leak correction (Nov 15, evening)

The data agent found that data/gate0b/train_text.json (the v1/v2 training text) contained 433 lines identical to
eval/text lines after normalisation (lowercase; letters, digits, apostrophes, spaces), about 330 of them from the
test splits ("fire it up", "halt", "begin cooking" and the like). Every training text from v3 on drops such lines
(sw/textfilter.py). The frozen v2 models were re-scored on the test rows whose text is not in train_text.json
(same dev threshold; `eval/gate0/text_bound.py --train`; raw: reports/gate0/text_bound_0b_*_knn_neg_nonoverlap.json):

| Model, `knn_neg` + expansion | Fan unseen (all 312 / non-overlap 242) | Fan rejection (all / non-overlap) | Microwave unseen (all 208 / non-overlap 173) | Microwave rejection (all / non-overlap) |
|---|---|---|---|---|
| bge-small tuned v2 | 82.7% / 81.0% | 93.8% / 93.8% | 92.3% / 91.3% | 90.1% / 90.0% |
| bge-base tuned v2 | 82.7% / 81.0% | 95.7% / 95.6% | 90.4% / 88.4% | 93.2% / 93.1% |

So the leak inflated unseen accuracy by about 1 to 2 points; rejection is unchanged.

## Teacher v3 (frozen, Nov 15, 6:35 PM)

bge-small-en-v1.5 fine-tuned on 72 training appliances (40 new, with swing, rotate, tilt, patrol and other motion
and toggle modes), leak-filtered text, selected on held-out training products (docs/decisions.md). Eval test, run
once (raw: reports/gate0/text_bound_teacher_v3_*.json; none of the test lines is in its training text):

| Matcher (+ expansion) | Fan unseen / rejection | Microwave unseen / rejection |
|---|---|---|
| `knn_neg` | 85.6% / 93.2% | 87.5% / 90.1% |
| `linear` | 84.0% / 91.4% | 79.8% / 94.4% |

Against leak-free v2 bge-small (`knn_neg`): fan +4.6 points unseen (the motion products helped oscillate), microwave
-3.8 points. Still short of 90 / 95 on both; this is the teacher the audio encoder learns from.
