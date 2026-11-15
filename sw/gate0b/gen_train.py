"""Gate 0b training text: a local LLM (Ollama) writes many phrasings for every action of
the training products in products.json, plus non-commands per product. Fans and
microwaves are excluded from the products; nothing in eval/ is read.

Output: data/gate0b/train_text.json  {product: {"actions": {name: [..]}, "none": [..]}}
Resumable: products already in the output file are skipped.
"""
import json, re, sys, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/gate0b/train_text.json"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5:32b-instruct-q4_K_M"
N = 45


def ask(prompt, temp=0.95):
    body = json.dumps({"model": MODEL, "prompt": prompt, "stream": False,
                       "options": {"temperature": temp, "num_predict": 1500}}).encode()
    req = urllib.request.Request("http://localhost:11434/api/generate", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)["response"]


def lines(txt):
    out = []
    for l in txt.splitlines():
        l = re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", l).strip().strip('"').strip().lower().rstrip(".!")
        if 1 <= len(l.split()) <= 14 and not l.endswith(":"):
            out.append(l)
    return sorted(set(out))


def main():
    prods = {k: v for k, v in json.load(open(Path(__file__).with_name("products.json"))).items() if not k.startswith("_")}
    done = json.load(open(OUT)) if OUT.exists() else {}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for p, acts in prods.items():
        if p in done:
            continue
        menu = "; ".join(f"{a} ({d})" for a, d in acts.items())
        rec = {"actions": {}, "none": []}
        for a, d in acts.items():
            prompt = (f"A voice-controlled {p} understands these actions: {menu}.\n"
                      f"Write {N} different things a person at home might say to the {p} to mean exactly: {d}.\n"
                      "Mix: very short commands (one to three words), casual and slang, polite requests and questions, "
                      "and indirect statements that imply it (for example a complaint about how things are now). "
                      "Use varied numbers, times and levels where the action takes a value. "
                      "Each line must clearly mean this action and not any of the others. "
                      "One per line, lowercase, no numbering, no quotes, nothing else.")
            rec["actions"][a] = lines(ask(prompt))
        prompt = (f"Write {N} different things someone might say in a room with a voice-controlled {p} that are NOT instructions to it: "
                  f"small talk, remarks that mention the {p} or its settings without asking for anything, TV or radio lines, "
                  "and commands meant for other devices. One per line, lowercase, no numbering, no quotes, nothing else.")
        rec["none"] = lines(ask(prompt))
        done[p] = rec
        json.dump(done, open(OUT, "w", encoding="utf8"), indent=0)
        n = sum(len(v) for v in rec["actions"].values())
        print(f"{p}: {n} commands, {len(rec['none'])} none ({time.time()-t0:.0f}s)", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
