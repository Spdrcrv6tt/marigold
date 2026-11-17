"""SDK phrase expansion: a local LLM (Ollama) turns each action's description and typed examples into many
paraphrases, and writes a generic non-command set for reject calibration. It sees only the config, never any eval/
file.

Review K5 (Nov 17): any product (one entry per config file, keyed by the file stem, which is the key compile.py looks
up); the default output is the file compile.py reads by default (data/expand/expanded_qwen25.json); a paraphrase the
LLM writes under two or more actions of one product, or that equals another action's typed example, is dropped.

Usage: python sdk/expand.py [CONFIG.json ...] [--model qwen2.5:32b-instruct-q4_K_M] [--out data/expand/expanded_qwen25.json]
       (no configs: every sdk/configs/*.json)
"""
import argparse, json, re, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data/expand/expanded_qwen25.json"          # the file sdk/compile.py reads by default
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


def expand_config(c, model, per, ask_fn=None, log=print):
    """One product: {action: [paraphrases]} and [non-commands]. ask_fn(model, prompt) -> text (tests pass a stub)."""
    ask_fn = ask_fn or ask
    prod = c["product"].split("(")[0].strip().lower()
    acts = c["actions"]
    others = "; ".join(f"{x['name']}: {x['description']}" for x in acts)
    raw = {}
    for act in acts:
        got = []
        for st in STYLES:
            prompt = (f"A voice-controlled {prod} understands these actions: {others}.\n"
                      f"Write {per} different things a person might say to the {prod} to mean exactly this action: "
                      f"{act['description']} (action '{act['name']}').\n"
                      f"Examples the maker typed: {'; '.join(act['examples'])}.\n"
                      f"Focus on {st}. Do not repeat the examples. "
                      + ("Include varied values (numbers, times, levels) where the action takes one. " if act.get("values") else "")
                      + "Make sure each line could only mean this action, not one of the others. "
                      "One per line, lowercase, no numbering, no quotes, nothing else.")
            got += lines(ask_fn(model, prompt))
        raw[act["name"]] = set(got)
    typed = {act["name"]: {e.lower() for e in act["examples"]} for act in acts}
    count = {}
    for name, got in raw.items():
        for g in got:
            count[g] = count.get(g, 0) + 1
    res = {}
    for name, got in raw.items():
        foreign = set().union(*[typed[o] for o in typed if o != name]) if len(typed) > 1 else set()
        res[name] = sorted(g for g in got if g not in typed[name] and count[g] == 1 and g not in foreign)
        log(f"{c.get('product')} {name}: {len(res[name])} (dropped {len(got) - len(res[name])})")
    neg = []
    for np_ in NEG_PROMPTS:
        prompt = (f"Write {per} different short things ({np_.format(product=prod)}). "
                  f"None of them may be an instruction to a {prod}. One per line, lowercase, no numbering, no quotes, nothing else.")
        neg += lines(ask_fn(model, prompt))
    all_cmd = set().union(*res.values(), *typed.values())
    return res, sorted(set(neg) - all_cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="*", help="config files (default: every sdk/configs/*.json)")
    ap.add_argument("--model", default="qwen2.5:32b-instruct-q4_K_M")
    ap.add_argument("--per", type=int, default=40)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    a = ap.parse_args()
    cfgs = [Path(x) for x in a.configs] or sorted((ROOT / "sdk/configs").glob("*.json"))
    res, neg = {}, {}
    t0 = time.time()
    for f in cfgs:
        c = json.load(open(f, encoding="utf8"))
        res[f.stem], neg[f.stem] = expand_config(c, a.model, a.per, log=lambda s: print(f"{s} ({time.time() - t0:.0f}s)", flush=True))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"generator": a.model, "actions": res, "negatives": neg}, open(a.out, "w", encoding="utf8"), indent=1)
    print("done", a.out)


if __name__ == "__main__":
    main()
