"""Held-out guard for every training text (embedder and audio).

Rule (coordinator, Nov 15, 2026): drop any training line whose normalized text (lowercase;
letters, digits, apostrophes and spaces; runs of spaces collapsed) equals any line of
eval/text/paraphrases.txt or eval/text/noncommands.txt, in either split.
Iteration 2 (Nov 16): also every line of the blind test set eval/text_v2/ (TEST-ONLY), whatever its file layout
(.txt lines with an optional " | values" tail, .jsonl/.json "text" fields).
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def norm(s):
    return " ".join(re.sub(r"[^a-z0-9' ]", " ", s.lower()).split())


def eval_lines():
    out = set()
    for f in ("paraphrases.txt", "noncommands.txt"):
        for line in open(ROOT / "eval/text" / f, encoding="utf8"):
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            out.add(norm(line.partition(" | ")[0]))
    v2 = ROOT / "eval/text_v2"
    for f in sorted(v2.rglob("*")) if v2.is_dir() else []:
        if f.suffix in (".txt", ".tsv", ".md"):
            for line in open(f, encoding="utf8"):
                line = line.rstrip("\n")
                if line and not line.startswith("#"):
                    out.add(norm(line.partition(" | ")[0].split("\t")[0]))
        elif f.suffix in (".jsonl", ".json"):
            txt = f.read_text(encoding="utf8")
            recs = [json.loads(l) for l in txt.splitlines() if l.strip()] if f.suffix == ".jsonl" else [json.loads(txt)]
            stack = list(recs)
            while stack:
                x = stack.pop()
                if isinstance(x, dict):
                    if isinstance(x.get("text"), str):
                        out.add(norm(x["text"]))
                    stack += [v for v in x.values() if isinstance(v, (dict, list))]
                elif isinstance(x, list):
                    stack += [v for v in x if isinstance(v, (dict, list))]
                    out.update(norm(v) for v in x if isinstance(v, str))
    out.discard("")
    return out


class EvalFilter:
    def __init__(self):
        self.ev = eval_lines()
        self.dropped = 0

    def ok(self, text):
        bad = norm(text) in self.ev
        self.dropped += bad
        return not bad

    def keep(self, texts):
        return [t for t in texts if self.ok(t)]
