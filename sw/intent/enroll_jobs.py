"""SDK audio enrollment, step 1: TTS jobs for a product's config phrases (typed examples + SDK expansion) and the
expansion's non-commands, spoken by TRAIN voices only (3 Kokoro train voices), for data/tts/jobs_enroll.jsonl.
Lines identical to an eval/text line are dropped (sw/textfilter.py). Then:
  python -I sw/data/tts_build.py synth data/tts/jobs_enroll.jsonl kokoro 0 1 cuda
The chip model turns these clips into audio-space prototypes (sw/intent/evaluate.py --enroll). The model weights
never see them; this is OEM-side enrollment from typed text, like the text prototypes.
"""
import json, random, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sw"))
from textfilter import EvalFilter  # noqa: E402

VOICES = ["af_bella", "am_adam", "bf_isabella"]


def main():
    flt = EvalFilter()
    X = json.load(open(ROOT / "data/expand/expanded_qwen25.json", encoding="utf8"))
    tmpl = json.loads(open(ROOT / "data/tts/jobs_train.jsonl", encoding="utf8").readline())
    rng = random.Random(7); jobs = []; seen = set()

    def add(text, prod, action, cat):
        if not flt.ok(text):
            return
        for v in VOICES:
            if (text, v) in seen:
                continue
            seen.add((text, v))
            slug = f"enroll-{prod}-{action or 'none'}"; i = len(jobs)
            j = dict(tmpl)
            j.update(id=f"{slug}-{i:05d}-kokoro-{v.replace('_', '-')}", text=text, product=prod, action=action, values={},
                     category=cat, set="enroll", split="train", voice=f"kokoro/{v}", engine="kokoro", speaker=v,
                     locale="en-GB" if v.startswith("b") else "en-US", length_scale=round(rng.uniform(0.92, 1.15), 3),
                     out=f"enroll/wav/{slug}/{slug}-{i:05d}-kokoro-{v.replace('_', '-')}.wav")
            jobs.append(j)
    for prod in ("fan", "microwave"):
        c = json.load(open(ROOT / f"sdk/configs/{prod}.json", encoding="utf8"))
        for a in c["actions"]:
            for t in list(dict.fromkeys(a["examples"] + X["actions"][prod].get(a["name"], []))):
                add(t, prod, a["name"], None)
        for t in X.get("negatives", {}).get(prod, []):
            add(t, prod, None, "none")
    with open(ROOT / "data/tts/jobs_enroll.jsonl", "w", encoding="utf8") as f:
        for j in jobs:
            f.write(json.dumps(j) + "\n")
    print(len(jobs), "jobs; eval-identical dropped", flt.dropped)


if __name__ == "__main__":
    main()
