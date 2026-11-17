"""Iteration-2 text sets written from the two configs only (sdk/configs/fan.json, microwave.json), by local LLMs:

  dev2  eval/text_dev2/  calibration DEV set for the reject fix (qwen2.5:14b): 40 phrasings per action incl. long,
        chatty and indirect ones; non-commands: chat 100 (incl. short fillers), near 50, tv 40, other_device 60.
  v3    eval/text_v3/    blind TEST set, TEST-ONLY, same balance as eval/text_v2 (60 per action; chat 130, near 60,
        tv 50, other_device 40), a different generator (gemma4:12b), prompts and seed. Written before any result of
        the reject fix is looked at; scored once.

Neither generator sees eval/text, eval/text_v2 or any training text. Lines identical (normalised) to eval/text,
eval/text_v2, the other new set, the config examples or (v3 only) the training text are dropped.
Usage: python sw/teacher4/gen_evalsets.py dev2|v3
"""
import json, random, re, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sw"))
from textfilter import norm  # noqa: E402

SETS = {
    "dev2": dict(model="qwen2.5:14b", seed=2026111602, per_action=40, non={"chat": 100, "near": 50, "tv": 40, "other_device": 60}, out="eval/text_dev2"),
    "v3": dict(model="gemma4:12b", seed=2026111603, per_action=60, non={"chat": 130, "near": 60, "tv": 50, "other_device": 40}, out="eval/text_v3"),
}
SLOTRULE = {"fan": "speed is a level 1 to 5 (low=1, medium=3, high=5)",
            "microwave": "time is in seconds (two minutes = 120), power is a level 1 to 10 (low=3, medium or half=5, high or full=10)"}


def ask(model, prompt, seed, temp=1.0):
    req_ = {"model": model, "prompt": prompt, "stream": False, "keep_alive": "10m",
            "options": {"temperature": temp, "seed": seed, "num_predict": 3000, "num_ctx": 4096}}
    if model.startswith("gemma4"):
        req_["think"] = False          # gemma4 thinks by default; the answer alone is wanted
    body = json.dumps(req_).encode()
    req = urllib.request.Request("http://localhost:11434/api/generate", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)["response"]


def clean(l):
    l = re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", l).strip().strip('"').strip()
    text, _, vals = l.partition("|")
    text = re.sub(r"[.!?]+$", "", text.strip().lower().replace("’", "'")).strip()
    return text, vals.strip()


def menu(c):
    return "; ".join(f"{a['name']}: {a['description']}" for a in c["actions"])


def act_prompt(p, c, a, n, variant):
    slots = a.get("values") or []
    vs = (f" The action takes values ({', '.join(slots)}): {SLOTRULE[p]}. After every line that states a value, write ' | ' "
          f"and the values as slot=number joined by ';' (for example 'make it about three | {slots[0]}=3'). A line may leave "
          "the value out only if the action allows it.") if slots else " Do not add anything after the sentence."
    style = ["Mix many styles: one to three word commands, casual and slang, polite requests, questions, indirect statements "
             "that imply it (a complaint about how things are now), and long chatty sentences with fillers like 'um', "
             "'so yeah', 'you know what', 'okay so'.",
             "Vary them widely: terse orders, friendly asks, sentences that start with a reason ('it's freezing so...'), "
             "hedged requests ('could you maybe...'), two-part sentences, rambling ones with 'like' and 'uh', and indirect "
             "hints. About a third should be longer than eight words."][variant]
    return (f"A voice-controlled {p} understands exactly these actions: {menu(c)}.\n"
            f"Write {n} different things a person at home might say to the {p}, after its wake word, that mean exactly the "
            f"action '{a['name']}' ({a['description']}) and none of the others. {style}{vs}\n"
            "One per line, lowercase, no numbering, no quotes, no trailing punctuation, nothing else.")


NON = {
    "chat": ["Write {n} short things people say at home that are not instructions to any device: small talk, reactions and "
             "fillers. At least a third must be one to three words, like interjections, hesitations, thanks, apologies, "
             "agreements and refusals.",
             "Write {n} varied lines of everyday conversation in a kitchen or living room, said to another person, not to a "
             "device: replies, short reactions, filler words, questions about plans, comments on food or weather. Many very "
             "short."],
    "near": ["Write {n} things someone might say near a voice-controlled {p} that mention it, its parts or its settings but do "
             "NOT ask it to do anything: past events, opinions, facts, plans for later, questions to another person.",
             "Write {n} remarks a family member might make about the {p} (how it works, when it was bought, what it did "
             "yesterday, whether it is loud, what setting someone likes) that are not requests to it."],
    "tv": ["Write {n} lines that could come out of a television or radio playing in the room: news, ads, sports commentary, "
           "drama dialogue, weather, game shows.",
           "Write {n} snippets of TV and radio audio: presenters, commercials, sitcom lines, cooking shows, sports calls, "
           "traffic reports."],
    "other_device": ["Write {n} voice commands meant for OTHER devices in a home, never a fan and never a microwave: TV, "
                     "lights, thermostat, speaker, door lock, blinds, vacuum, oven, coffee maker. Name or clearly imply the "
                     "other device.",
                     "Write {n} things people tell smart home gadgets other than a fan or a microwave: music, lamps, heating, "
                     "TV, alarms, locks, robot vacuum, kettle. Each must clearly be for that other gadget."],
}


def main():
    name = sys.argv[1]
    S = SETS[name]
    rng = random.Random(S["seed"])
    variant = 0 if name == "dev2" else 1
    cfgs = {p: json.load(open(ROOT / f"sdk/configs/{p}.json", encoding="utf8")) for p in ("fan", "microwave")}
    banned = set()
    for d in ("eval/text", "eval/text_v2", "eval/text_dev2" if name == "v3" else "eval/text_v3"):
        for f in (ROOT / d).glob("*.txt") if (ROOT / d).is_dir() else []:
            for line in open(f, encoding="utf8"):
                if line.strip() and not line.startswith("#"):
                    banned.add(norm(line.partition(" | ")[0]))
    banned |= {norm(e) for c in cfgs.values() for a in c["actions"] for e in a["examples"]}
    if name == "v3":
        tt = json.load(open(ROOT / "data/gate0b/train_text_v4.json", encoding="utf8"))
        banned |= {norm(t) for r in tt.values() for v in list(r["actions"].values()) + [r["none"]] for t in v}
    jobs = []
    for p, c in cfgs.items():
        for a in c["actions"]:
            for k in range(2):
                jobs.append((("act", p, a["name"]), act_prompt(p, c, a, int(S["per_action"] * 0.8), (variant + k) % 2), rng.randrange(1 << 30)))
    for cat, n in S["non"].items():
        for k in range(2):
            for p in (["fan", "microwave"] if cat == "near" else ["fan"]):
                q = NON[cat][(variant + k) % 2].format(n=int(n * 0.8), p=p)
                jobs.append((("non", p, cat), q + "\nOne per line, lowercase, no numbering, no quotes, nothing else.", rng.randrange(1 << 30)))
    with ThreadPoolExecutor(4) as ex:
        outs = list(ex.map(lambda j: (j[0], ask(S["model"], j[1], j[2])), jobs))
    acts, non = {}, {}
    for (kind, p, x), txt in outs:
        for l in txt.splitlines():
            t, vals = clean(l)
            if not t or len(t.split()) > 25 or norm(t) in banned or re.search(r"[^\x00-\x7f]", t):
                continue
            if kind == "act":
                acts.setdefault((p, x), {})[norm(t)] = (t, vals)
            else:
                non.setdefault(x, {})[norm(t)] = t
    out = ROOT / S["out"]; out.mkdir(parents=True, exist_ok=True)
    head = (f"# marigold {name}: written {'as a DEV set (thresholds may be set on it)' if name == 'dev2' else 'blind, TEST-ONLY: never train, tune or set thresholds on it'}"
            f"\n# generated 2026-11-16 by local {S['model']} (seed {S['seed']}) from sdk/configs/fan.json and microwave.json only (sw/teacher4/gen_evalsets.py)"
            "\n# lines identical to eval/text, eval/text_v2, the other new set, config examples" + (" or training text" if name == "v3" else "") + " were dropped\n")
    seen_all = set()
    with open(out / "paraphrases.txt", "w", encoding="utf8") as f:
        f.write(head + "# format: '## <product> <action>' blocks; ' | slot=value' (time in seconds, speed 1-5, power 1-10)\n")
        for p, c in cfgs.items():
            for a in c["actions"]:
                slots = a.get("values") or []
                rows = []
                for key, (t, vals) in sorted(acts.get((p, a["name"]), {}).items()):
                    if key in seen_all:
                        continue
                    kv = dict(x.split("=", 1) for x in vals.replace(" ", "").split(";") if "=" in x)
                    kv = {k: v for k, v in kv.items() if k in slots and re.fullmatch(r"\d+", v)}
                    if slots and a["name"] != "start" and not kv:
                        continue
                    if any((k == "speed" and not 1 <= int(v) <= 5) or (k == "power" and not 1 <= int(v) <= 10) for k, v in kv.items()):
                        continue
                    rows.append(t + (" | " + ";".join(f"{k}={v}" for k, v in kv.items()) if kv else ""))
                rng.shuffle(rows)
                rows = rows[:S["per_action"]]
                seen_all |= {norm(r.partition(" | ")[0]) for r in rows}
                f.write(f"## {p} {a['name']}\n" + "".join(r + "\n" for r in rows))
                print(p, a["name"], len(rows), flush=True)
    with open(out / "noncommands.txt", "w", encoding="utf8") as f:
        f.write(head + "# format: '## <block>' blocks (chat, near, tv, other_device); no line may trigger a fan or microwave action\n")
        for cat, n in S["non"].items():
            rows = [t for key, t in sorted(non.get(cat, {}).items()) if key not in seen_all]
            rng.shuffle(rows)
            rows = rows[:n]
            seen_all |= {norm(r) for r in rows}
            f.write(f"## {cat}\n" + "".join(r + "\n" for r in rows))
            print(cat, len(rows), flush=True)
    urllib.request.urlopen(urllib.request.Request("http://localhost:11434/api/generate",
                           json.dumps({"model": S["model"], "keep_alive": 0}).encode(), {"Content-Type": "application/json"})).read()


if __name__ == "__main__":
    main()
