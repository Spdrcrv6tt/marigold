"""Held-out guard for every training text (embedder and audio).

Rule (coordinator, Nov 15, 2026): drop any training line whose normalized text (lowercase;
letters, digits, apostrophes and spaces; runs of spaces collapsed) equals any line of
eval/text/paraphrases.txt or eval/text/noncommands.txt, in either split.
"""
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
