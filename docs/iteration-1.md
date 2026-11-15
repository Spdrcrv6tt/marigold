# Iteration 1: definition of done

Iteration 1 is done when Marigold meets all 12 targets below, first in software (golden integer model) and then bit for bit in RTL and on the KV260, the same standard Poinsetta met. The targets come from PLAN.md (John, Nov 15, 2026). They are starting targets: revise them, with a dated entry in docs/decisions.md, once the first software baseline exists. A missed target is reported as a finding with its measured number, never dropped.

| # | Item | Target | How it is measured | Gate phase | Status |
|---|---|---|---|---|---|
| 1 | Intent accuracy, phrasings the OEM typed | ≥ 97% | Held-out recordings (new speakers) of the config example phrases | 0 | open |
| 2 | Intent accuracy, phrasings the OEM never typed | ≥ 90% | Paraphrase test set (eval/text/paraphrases.txt, split=test; never trained on) | 0 | open |
| 3 | Value extraction | ≥ 95% exact | Numbers, durations, levels: "two minutes", "set it to 72", "level 3" | 0 | open |
| 4 | Rejecting non-commands after a wake | ≥ 95% rejected | Ordinary talk after the wake word fires nothing (eval/text/noncommands.txt, split=test) | 0 | open |
| 5 | Wake word detection | ≥ 95% at 1 to 3 m | Recorded wake phrases, several speakers and rooms | 1 | open |
| 6 | False wakes | ≤ 1 per 24 h | 24 h of TV, music and conversation | 1 | open |
| 7 | Response time | ≤ 300 ms median | End of speech to message on the output pins | 3 | open |
| 8 | Model size | ≤ 5M weights at 4-bit | ≤ 20 Mbit of ROM as built (packed image, tables included), about 1.7 mm² at 28 nm | 0 | open |
| 9 | Listening power, chip only | ≤ 100 µW at 28 nm (estimate) | Poinsettia's method: GF180 activity-based power, 45 nm layout, scaled to 28 nm; island includes the wake spotter | 4 | open |
| 10 | Product noise | ≤ 5 points accuracy loss | Fan at full speed or microwave running, 10 dB below the voice | 0 | open |
| 11 | Same silicon, two products | Fan and microwave configs on one model | sdk/configs/fan.json and microwave.json on one set of weights, no retraining | 0 | open |
| 12 | Bit-exact hardware | RTL matches the golden model | Verilator, mic pin to output pins, with negative controls that must fail | 2 | open |

## Rules for measuring

- **Held-out means held out.** The paraphrase and non-command test splits are evaluation only: never used for training, threshold tuning, example writing or prompt writing. Tuning uses the dev splits. Test-set speakers and TTS voices are disjoint from training ones.
- **Accuracy counts rejections as errors.** Targets 1 and 2 are scored with the same reject threshold that target 4 uses, chosen on dev. Reporting accuracy with rejection switched off is allowed only as a labeled side number.
- **Text first.** Gate 0 (text-only matching, no audio) bounds targets 2 and 4 from above. If text matching cannot reach them, audio will not either.
- **Real recordings beat synthetic.** Synthetic TTS audio is a stand-in until real recordings exist; any number measured only on synthetic audio says so.
- **Licenses.** SLURP and Fluent Speech Commands audio may be used for evaluation only (docs/licenses.md).
