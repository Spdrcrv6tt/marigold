"""Phone set, CMUdict lexicon (BSD-2-Clause style, docs/licenses.md) and the v1 fallback G2P.

Phone ids: 0 = CTC blank, 1..39 = the 39 CMUdict phones below (stress removed).
"""
import re
from functools import lru_cache
from pathlib import Path

R = Path(__file__).resolve().parents[2]
CMUDICT = R / "sw/wake/lex/cmudict.dict"          # in the repo since Nov 17, 2026 (was data/wake/lex, git-ignored)
PHONES = ["AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "EH", "ER", "EY", "F", "G", "HH", "IH", "IY",
          "JH", "K", "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH", "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH"]
PID = {p: i + 1 for i, p in enumerate(PHONES)}
N_OUT = len(PHONES) + 1


@lru_cache(maxsize=1)
def cmudict():
    d = {}
    for line in open(CMUDICT, encoding="utf-8"):
        line = line.split("#")[0].strip()
        if not line:
            continue
        w, *ph = line.split()
        w = re.sub(r"\(\d+\)$", "", w)
        seq = tuple(PID[re.sub(r"\d", "", p)] for p in ph)
        d.setdefault(w, [])
        if seq not in d[w]:
            d[w].append(seq)
    return d


def words(text):
    t = text.lower().replace("’", "'")
    t = re.sub(r"[^a-z' ]+", " ", t)
    return [w.strip("'") for w in t.split() if w.strip("'")]


def word_prons(w):
    """All CMUdict pronunciations of w, else the fallback (one guess). Returns (list, source)."""
    d = cmudict()
    if w in d:
        return d[w], "cmudict"
    # compound: two dictionary words
    for i in range(len(w) - 2, 1, -1):
        a, b = w[:i], w[i:]
        if a in d and b in d:
            return [d[a][0] + d[b][0]], "compound"
    return [letters_to_sound(w)], "rules"


# ---- crude letter-to-sound fallback (v1): the SDK warns and asks for a respelling when used
_LTS = [("tch", "CH"), ("sch", "S K"), ("ph", "F"), ("sh", "SH"), ("ch", "CH"), ("th", "TH"), ("ng", "NG"), ("ck", "K"),
        ("qu", "K W"), ("wh", "W"), ("ee", "IY"), ("ea", "IY"), ("oo", "UW"), ("ou", "AW"), ("ow", "OW"), ("oa", "OW"),
        ("ai", "EY"), ("ay", "EY"), ("ey", "EY"), ("ie", "IY"), ("oi", "OY"), ("oy", "OY"), ("au", "AO"), ("aw", "AO"),
        ("er", "ER"), ("ir", "ER"), ("ur", "ER"), ("ar", "AA R"), ("or", "AO R"), ("a", "AE"), ("e", "EH"), ("i", "IH"),
        ("o", "AA"), ("u", "AH"), ("y", "IY"), ("b", "B"), ("c", "K"), ("d", "D"), ("f", "F"), ("g", "G"), ("h", "HH"),
        ("j", "JH"), ("k", "K"), ("l", "L"), ("m", "M"), ("n", "N"), ("p", "P"), ("q", "K"), ("r", "R"), ("s", "S"),
        ("t", "T"), ("v", "V"), ("w", "W"), ("x", "K S"), ("z", "Z")]


def letters_to_sound(w):
    out, i = [], 0
    w = re.sub(r"(.)\1", r"\1", w)              # doubled letters
    if len(w) > 2 and w.endswith("e") and w[-2] not in "aeiou":
        w = w[:-1]                               # silent final e
    while i < len(w):
        for g, p in _LTS:
            if w.startswith(g, i):
                out += [PID[x] for x in p.split()]
                i += len(g)
                break
        else:
            i += 1
    return tuple(out)


def phrase_prons(text, max_variants=4):
    """Pronunciation variants of a phrase: the cartesian product of word variants, first ones first,
    capped at max_variants. Returns (list of tuples, notes)."""
    seqs, notes = [()], []
    for w in words(text):
        prons, src = word_prons(w)
        if src != "cmudict":
            notes.append(f"'{w}' not in CMUdict: {src} guess {' '.join(PHONES[p - 1] for p in prons[0])}")
        seqs = [s + p for s in seqs for p in prons][:max_variants]
    return seqs, notes


def utterance_phones(text):
    """First pronunciation of every word, or None when any word is not in CMUdict (training data)."""
    d = cmudict()
    out = []
    for w in words(text):
        if w not in d:
            return None
        out += d[w][0]
    return out
