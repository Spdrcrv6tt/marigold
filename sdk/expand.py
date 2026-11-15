"""SDK phrase expansion v0: a local LLM (Ollama) turns each action's description and
typed examples into many paraphrases, and writes a generic non-command set for
reject calibration. It sees only the config, never any eval/ file.

Usage: python sdk/expand.py --model qwen2.5:32b-instruct-q4_K_M --out data/expand/expanded.json
"""
import argparse, json, re, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STYLES = [
    "short, casual commands people actually say out loud, many only two to four words",
    "polite or wordy requests, and questions like 'could you ...'",
    "indirect ways: complaints or statements that imply the action without naming it",
    "slang, informal and regional ways, as an American adult might talk at home",
]
NEG_PROMPTS = [
    "everyday things people say at home to each other: chit-chat, questions, plans, small talk",
    "lines heard on TV or radio: news, sports commentary, ads, cooking shows, drama dialogue",
    "voice commands meant for other devices or a voice assistant: lights, TV, music, thermostat, timers, calendar, shopping lists, other appliances",
    "remarks that mention a {product} or the kitchen or home appliances but are not instructions to it",
]


def ask(model, prompt, temp=0.9):
    body = json.dumps({"model": model, "prompt": prompt, "stream": False,
                       "options": {"temperature": temp, "num_predict": 2048}}).encode()
    req = urllib.request.Request("http://localhost:11434/api/generate", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)["response"]


def lines(txt):
    out = []
    for l in txt.splitlines():
        l = re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", l).strip().strip('"').strip()
        l = l.lower().rstrip(".!")
        if 1 <= len(l.split()) <= 14 and not l.endswith(":"):
            out.append(l)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:32b-instruct-q4_K_M")
    ap.add_argument("--per", type=int, default=40)
    ap.add_argument("--out", default=str(ROOT / "data/expand/expanded.json"))
    a = ap.parse_args()
    res, neg = {}, {}
    t0 = time.time()
    for p in ("fan", "microwave"):
        c = json.load(open(ROOT / f"sdk/configs/{p}.json", encoding="utf8"))
        prod = c["product"].split("(")[0].strip().lower()
        acts = c["actions"]
        others = "; ".join(f"{x['name']}: {x['description']}" for x in acts)
        res[p] = {}
        for act in acts:
            got = []
            for st in STYLES:
                prompt = (f"A voice-controlled {prod} understands these actions: {others}.\n"
                          f"Write {a.per} different things a person might say to the {prod} to mean exactly this action: "
                          f"{act['description']} (action '{act['name']}').\n"
                          f"Examples the maker typed: {'; '.join(act['examples'])}.\n"
                          f"Focus on {st}. Do not repeat the examples. "
                          + ("Include varied values (numbers, times, levels) where the action takes one. " if act["values"] else "")
                          + "Make sure each line could only mean this action, not one of the others. "
                          "One per line, lowercase, no numbering, no quotes, nothing else.")
                got += lines(ask(a.model, prompt))
            ex = {e.lower() for e in act["examples"]}
            res[p][act["name"]] = sorted({g for g in got if g not in ex})
            print(f"{p} {act['name']}: {len(res[p][act['name']])} ({time.time()-t0:.0f}s)", flush=True)
        got = []
        for np_ in NEG_PROMPTS:
            prompt = (f"Write {a.per} different short things ({np_.format(product=prod)}). "
                      f"None of them may be an instruction to a {prod}. One per line, lowercase, no numbering, no quotes, nothing else.")
            got += lines(ask(a.model, prompt))
        neg[p] = sorted(set(got))
        print(f"{p} negatives: {len(neg[p])} ({time.time()-t0:.0f}s)", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"generator": a.model, "actions": res, "negatives": neg}, open(a.out, "w", encoding="utf8"), indent=1)
    print("done", a.out)


if __name__ == "__main__":
    main()
