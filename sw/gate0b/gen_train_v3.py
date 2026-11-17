"""Teacher v3 training text. Same prompts as gen_train.py (local Ollama, qwen2.5:32b) for the
products in products_v3.json, run on 4 threads; the 32 gate-0b products are reused from
train_text.json. Every product also gets a second batch of near-miss non-commands (lines that
mention the product or its settings without asking for anything).

Leak rule (Nov 15, coordinator): any line whose normalized text equals an eval/text line (either
split) is dropped here and again in every loader (sw/textfilter.py).

Nov 15, 5:40 PM: the 32B run spilled off the shared GPU into system RAM and was stopped after the 32 old
products; the new products were then generated with qwen2.5:14b (--model), a prioritized subset (--only).

Output: data/gate0b/train_text_v3.json  {product: {"actions": {name: [..]}, "none": [..]}}. Resumable.
"""
import json, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gate0b.gen_train as G  # noqa: E402
from gate0b.gen_train import lines, N  # noqa: E402
from textfilter import EvalFilter  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/gate0b/train_text_v3.json"
OLD = ROOT / "data/gate0b/train_text.json"


def ask(prompt):
    return G.ask(prompt)


def near_prompt(p, menu):
    return (f"A voice-controlled {p} understands these actions: {menu}.\n"
            f"Write {N} different things someone might say near the {p} that mention it, its parts or its settings "
            "but are NOT asking it to do anything: past events, opinions, stories, questions to another person, "
            "plans for later, facts about other households. Many should reuse words from the actions above. "
            "One per line, lowercase, no numbering, no quotes, nothing else.")


def act_prompt(p, menu, d):
    return (f"A voice-controlled {p} understands these actions: {menu}.\n"
            f"Write {N} different things a person at home might say to the {p} to mean exactly: {d}.\n"
            "Mix: very short commands (one to three words), casual and slang, polite requests and questions, "
            "and indirect statements that imply it (for example a complaint about how things are now). "
            "Use varied numbers, times and levels where the action takes a value. "
            "Each line must clearly mean this action and not any of the others. "
            "One per line, lowercase, no numbering, no quotes, nothing else.")


def none_prompt(p):
    return (f"Write {N} different things someone might say in a room with a voice-controlled {p} that are NOT instructions to it: "
            f"small talk, remarks that mention the {p} or its settings without asking for anything, TV or radio lines, "
            "and commands meant for other devices. One per line, lowercase, no numbering, no quotes, nothing else.")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:32b-instruct-q4_K_M")
    ap.add_argument("--only", default="", help="comma-separated products, in this order (default: all)")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    G.MODEL = a.model
    prods = {k: v for k, v in json.load(open(Path(__file__).with_name("products_v3.json"))).items() if not k.startswith("_")}
    if a.only:
        prods = {k: prods[k] for k in [x.strip() for x in a.only.split(",")]}
    old = json.load(open(OLD, encoding="utf8"))
    done = json.load(open(OUT, encoding="utf8")) if OUT.exists() else {}
    flt = EvalFilter()
    t0 = time.time()

    def one(p):
        acts = prods[p]
        menu = "; ".join(f"{a} ({d})" for a, d in acts.items())
        if p in old:
            rec = {"actions": dict(old[p]["actions"]), "none": list(old[p]["none"])}
        else:
            rec = {"actions": {a: lines(ask(act_prompt(p, menu, d))) for a, d in acts.items()},
                   "none": lines(ask(none_prompt(p)))}
        rec["none"] = sorted(set(rec["none"]) | set(lines(ask(near_prompt(p, menu)))))
        rec["actions"] = {a: flt.keep(v) for a, v in rec["actions"].items()}
        rec["none"] = flt.keep(rec["none"])
        return p, rec

    todo = [p for p in prods if p not in done]
    with ThreadPoolExecutor(a.threads) as ex:
        for p, rec in ex.map(one, todo):
            done[p] = rec
            json.dump(done, open(OUT, "w", encoding="utf8"), indent=0)
            n = sum(len(v) for v in rec["actions"].values())
            print(f"{p}: {n} commands, {len(rec['none'])} none ({time.time()-t0:.0f}s)", flush=True)
    print("done; eval-identical lines dropped:", flt.dropped, flush=True)


if __name__ == "__main__":
    main()
