"""Value head vocabulary, training targets and the deterministic parse (Marigold golden format, section 'values').

The value head is a per-frame CTC classifier over VOCAB on the encoder output. Greedy decoding: per frame the
lowest index with the largest logit; drop repeats, then blanks (id 0). parse() turns the token ids into values,
slot by slot, using only integer arithmetic (the firmware reimplements it line for line).
"""
import re

VOCAB = ["<b>",
         "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
         "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
         "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
         "hundred", "thousand", "a", "half", "quarter",
         "second", "minute", "hour", "degree", "percent",
         "level", "low", "medium", "high", "max", "min"]
ID = {w: i for i, w in enumerate(VOCAB)}
NUMV = {w: i - 1 for i, w in enumerate(VOCAB[1:21], 1)}                 # zero..nineteen -> 0..19
NUMV.update({w: 20 + 10 * k for k, w in enumerate(VOCAB[21:29])})      # twenty..ninety
UNIT_S = {"second": 1, "minute": 60, "hour": 3600}

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
         "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def num_words(n):
    """Integer 0..999999 -> American English words (no 'and')."""
    if n < 20:
        return [_ONES[n]]
    if n < 100:
        return [_TENS[n // 10]] + ([_ONES[n % 10]] if n % 10 else [])
    if n < 1000:
        return [_ONES[n // 100], "hundred"] + (num_words(n % 100) if n % 100 else [])
    return num_words(n // 1000) + ["thousand"] + (num_words(n % 1000) if n % 1000 else [])


def spoken_words(text):
    """Text as a TTS voice would say it, for targets: digits, times, %, degrees expanded; lowercase words."""
    t = text.lower().replace("%", " percent ").replace("°", " degrees ").replace("-", " ")
    t = re.sub(r"(\d+):(\d\d)", lambda m: f"{m.group(1)} {int(m.group(2))}" if m.group(2) != "00" else m.group(1), t)
    t = re.sub(r"(\d+)\.5\b", r"\1 and a half", t)
    t = re.sub(r"(\d+)(st|nd|rd|th)\b", r"\1", t)
    t = re.sub(r"\d+", lambda m: " ".join(num_words(int(m.group()))) if int(m.group()) < 10 ** 6 else m.group(), t)
    return re.findall(r"[a-z']+", t)


_MAP = {"seconds": "second", "secs": "second", "sec": "second", "minutes": "minute", "mins": "minute",
        "hours": "hour", "hrs": "hour", "hr": "hour", "degrees": "degree", "quarters": "quarter",
        "lowest": "min", "minimum": "min", "highest": "max", "maximum": "max", "full": "max",
        "mid": "medium", "middle": "medium", "an": "a"}
_LEVELW = {"level", "speed", "setting", "notch", "power"}
_LINK = {"to", "at", "on", "of", "number"}
_AFTER_A = {"second", "minute", "hour", "half", "quarter"}


def targets(text):
    """Token ids the value head should emit for this transcript (the value words only, in order)."""
    w = [_MAP.get(x, x) for x in spoken_words(text)]
    out = []
    for i, x in enumerate(w):
        nxt = w[i + 1] if i + 1 < len(w) else ""
        if x == "min" and i and w[i - 1] in NUMV:
            x = "minute"
        if x in _LEVELW:            # "level 3", "speed to 3", "power at 7": a level marker before a number word
            nx2 = w[i + 2] if i + 2 < len(w) else ""
            if nxt in NUMV or (nxt in _LINK and nx2 in NUMV):
                out.append(ID["level"])
        elif x == "a":
            if nxt in _AFTER_A:
                out.append(ID["a"])
        elif x in ID and x != "<b>":
            out.append(ID[x])
    return out


def ctc_greedy(ids):
    """Frame argmax ids -> token ids (drop repeats, then blanks)."""
    out, prev = [], -1
    for i in ids:
        if i != prev and i != 0:
            out.append(int(i))
        prev = i
    return out


def _number_run(ws):
    """List of number words -> integer (see golden-format.md, value parse step 1)."""
    total, group, prev, clock = 0, 0, "", False
    for w in ws:
        if w == "hundred":
            group = max(group, 1) * 100; prev = "H"; continue
        if w == "thousand":
            total += max(group, 1) * 1000; group = 0; prev = "K"; continue
        v = NUMV[w]
        kind = "o" if v < 10 else ("t" if v < 20 else "T")
        if prev == "":
            group = v
        elif prev == "H" or prev == "K":
            group += v
        elif kind == "o":
            group = group + v if (prev == "T" and group % 10 == 0) else group * 10 + v
        else:                               # teen or tens after a ones/teen/tens word: "three fifty" = 350
            group = group * 100 + v; clock = True
        prev = kind
    return total + group, clock and total == 0


def items(tokens):
    """Token ids -> list of (kind, value): ('dur', seconds), ('deg', n), ('pct', n), ('lvl', n), ('word', w), ('num', n)."""
    ws = [VOCAB[t] for t in tokens]
    seq, i = [], 0                            # numbers merged: ('n', value) or ('w', word)
    while i < len(ws):
        if ws[i] in NUMV or ws[i] in ("hundred", "thousand"):
            j = i
            while j < len(ws) and (ws[j] in NUMV or ws[j] in ("hundred", "thousand")):
                j += 1
            v, clk = _number_run(ws[i:j])
            seq.append(("n", v) if not clk else ("c", v)); i = j
        else:
            seq.append(("w", ws[i])); i += 1
    clocks = {j for j, x in enumerate(seq) if x[0] == "c"}
    seq = [("n", v) if k == "c" else (k, v) for k, v in seq]
    out, dur, last_unit, i = [], 0, 0, 0
    have_dur = False
    while i < len(seq):
        k, v = seq[i]
        nk, nv = seq[i + 1] if i + 1 < len(seq) else ("", None)
        n2k, n2v = seq[i + 2] if i + 2 < len(seq) else ("", None)
        if k == "w" and v == "a" and nk == "w" and nv in ("half", "quarter"):
            k, v = "w", nv; i += 1; nk, nv = n2k, n2v                       # "a half" == "half"
            n2k, n2v = seq[i + 2] if i + 2 < len(seq) else ("", None)
        if (k == "n" or (k == "w" and v == "a")) and nk == "w" and nv in UNIT_S:   # n unit, a unit
            u = UNIT_S[nv]; dur += (v if k == "n" else 1) * u; last_unit = u; have_dur = True; i += 2; continue
        if k == "n" and nk == "w" and nv == "a" and n2k == "w" and n2v in ("half", "quarter"):
            nk, nv = n2k, n2v; i += 1                                         # "two a half minute"
        if (k == "n" or (k == "w" and v == "a")) and nk == "w" and nv in ("half", "quarter"):
            # n + fraction + unit
            m = 2 if nv == "half" else 4
            n = v if k == "n" else 1
            j = i + 2
            if j < len(seq) and seq[j][0] == "w" and seq[j][1] == "a":
                j += 1
            if j < len(seq) and seq[j][0] == "w" and seq[j][1] in UNIT_S:
                u = UNIT_S[seq[j][1]]; dur += n * u + u // m; last_unit = u; have_dur = True; i = j + 1; continue
        if k == "w" and v in ("half", "quarter"):
            m = 2 if v == "half" else 4
            j = i + 1
            if j < len(seq) and seq[j][0] == "w" and seq[j][1] == "a":
                j += 1
            if j < len(seq) and seq[j][0] == "w" and seq[j][1] in UNIT_S:      # "half a minute"
                u = UNIT_S[seq[j][1]]; dur += u // m; last_unit = u; have_dur = True; i = j + 1; continue
            if last_unit:                                                     # "a minute and a half"
                dur += last_unit // m; i += 1; continue
            out.append(("word", "medium" if v == "half" else "quarter")); i += 1; continue
        if k == "n" and nk == "w" and nv == "degree":
            out.append(("deg", v)); i += 2; continue
        if k == "n" and nk == "w" and nv == "percent":
            out.append(("pct", v)); i += 2; continue
        if k == "w" and v == "level" and nk == "n":
            out.append(("lvl", nv)); i += 2; continue
        if k == "w" and v == "level" and nk == "w" and nv == "level":
            i += 1; continue
        if k == "n" and last_unit in (60, 3600) and (i == 0 or seq[i - 1][0] == "w" and seq[i - 1][1] in UNIT_S):
            dur += v * (last_unit // 60); last_unit = last_unit // 60; i += 1; continue   # "one minute thirty"
        if k == "n":
            out.append(("clk", v) if i in clocks else ("num", v)); i += 1; continue
        if v in ("low", "medium", "high", "max", "min"):
            out.append(("word", v))
        i += 1
    if have_dur:
        out.insert(0, ("dur", dur))
    return out


LEVEL_CODE = {"low": -1, "medium": -2, "high": -3, "max": -4, "min": -5}   # coordinator ruling, Nov 15
NO_VALUE = -(1 << 31)          # nothing heard for a slot (R16, John's ruling of Nov 17, 2026): the island leaves it out


def island_value(raw, slot):
    """What the island does with a raw slot value (architecture.md 8.7 + ruling of Nov 15): codes -1..-5 are level
    words, mapped with the slot entry's low/medium/high bytes (0 = default: low = min, medium = floor((min + max)
    / 2), high = max; max -> max, min -> min); then the value is clamped to [min, max]. NO_VALUE gives None: the
    slot is not in the host frame (R16), never clamped to the minimum."""
    if raw == NO_VALUE:
        return None
    lo, hi, w = slot["min"], slot["max"], slot.get("words") or {}
    if -5 <= raw <= -1:
        raw = {-1: w.get("low") or lo, -2: w.get("medium") or (lo + hi) // 2, -3: w.get("high") or hi,
               -4: hi, -5: lo}[raw]
    return min(max(raw, lo), hi)


def raw_values(tokens, slots):
    """One raw i32 per config slot, each slot extracted independently of the others and of the action (the core
    computes these before the island picks the action; the island clamps the action's slots to [min, max]).
    slots: list of dicts {type: 0 number | 1 duration (s) | 2 level, min, max, words?} in slot order.
      duration: the duration, else a clock reading (m:ss -> 60 m + ss), else the first bare number;
      level:    'level n', else the first level word as a code (-1 low, -2 medium, -3 high, -4 max, -5 min),
                else the first bare number;
      number:   the first degree or percent number, else the first bare number, else 'level n';
      nothing heard: NO_VALUE (INT32_MIN; no said value is negative below the level codes, so it is never a value)."""
    it = items(tokens)
    first = lambda kinds: next((x for x in it if x[0] in kinds), None)
    out = []
    for s in slots:
        t, lo, hi = s["type"], s["min"], s["max"]
        v = None
        if t == 1:
            x = first(("dur",)) or first(("clk",)) or first(("num",))
            v = None if x is None else (x[1] // 100 * 60 + x[1] % 100 if x[0] == "clk" else x[1])
        elif t == 2:
            x = first(("lvl",))
            if x:
                v = x[1]
            else:
                x = first(("word",))
                v = LEVEL_CODE.get(x[1]) if x else None
                if v is None:
                    x = first(("num", "pct"))
                    v = x[1] if x else None
        else:
            x = first(("deg", "pct")) or first(("num", "clk")) or first(("lvl",))
            v = x[1] if x else None
        out.append(NO_VALUE if v is None else int(v))
    return out


def parse(tokens, slots):
    """raw_values as the island turns them into host values (level codes mapped, clamped to [min, max]; None for a
    slot with nothing heard, which the host frame leaves out)."""
    return [island_value(v, s) for v, s in zip(raw_values(tokens, slots), slots)]


if __name__ == "__main__":
    tests = [("set it to 72", [{"type": 0, "min": 50, "max": 90}]),
             ("two and a half minutes", [{"type": 1, "min": 1, "max": 5999}]),
             ("a minute and a half", [{"type": 1, "min": 1, "max": 5999}]),
             ("half a minute", [{"type": 1, "min": 1, "max": 5999}]),
             ("one minute thirty", [{"type": 1, "min": 1, "max": 5999}]),
             ("heat it for 3 minutes on high", [{"type": 1, "min": 1, "max": 5999}, {"type": 2, "min": 1, "max": 10, "words": {"high": 10}}]),
             ("level two", [{"type": 2, "min": 1, "max": 5}]),
             ("speed four", [{"type": 2, "min": 1, "max": 5}]),
             ("put it on three", [{"type": 2, "min": 1, "max": 5}]),
             ("an hour", [{"type": 1, "min": 1, "max": 9999}]),
             ("set it to three fifty degrees", [{"type": 0, "min": 0, "max": 500}]),
             ("1:30", [{"type": 1, "min": 1, "max": 5999}]),
             ("go to medium", [{"type": 2, "min": 1, "max": 5}]),
             ("set the power to seven", [{"type": 1, "min": 1, "max": 5999}, {"type": 2, "min": 1, "max": 10}]),
             ("cook for 90 seconds at power level 8", [{"type": 1, "min": 1, "max": 5999}, {"type": 2, "min": 1, "max": 10}]),
             ("half power", [{"type": 2, "min": 1, "max": 10, "words": {"medium": 5}}]),
             ("go to medium speed", [{"type": 2, "min": 1, "max": 4}])]
    for t, sl in tests:
        tk = targets(t)
        print(f"{t!r:42} {[VOCAB[i] for i in tk]} -> {parse(tk, sl)}")
