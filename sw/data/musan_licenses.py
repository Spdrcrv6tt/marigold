#!/usr/bin/env python
"""Per-file license check for MUSAN noise and music (data/musan/musan). Keeps only public domain, CC0
and CC BY files (attribution recorded); rejects ShareAlike, NonCommercial, NoDerivs and anything unclear.

Writes data/noise/musan_clean.json (accepted, with type, pool split, license, attribution block) and
data/noise/musan_rejected.json (with reasons). MUSAN speech (LibriVox, US government) is not used.
Pool split: about 1 in 5 accepted files goes to eval, by a hash of the file id, so eval and train
never share a noise file.
"""
import hashlib, json, re
from pathlib import Path

R = Path(__file__).resolve().parents[2]
M = R / "data/musan/musan"
ID = re.compile(r"\b((?:music|noise)-[a-z-]+-\d{4})\b")


def classify(txt):
    t = txt.lower()
    if re.search(r"non-?commercial|\bnc\b|by-nc", t):
        return None, "non-commercial"
    if re.search(r"no ?deriv|by-nd|\bnd\b", t):
        return None, "no-derivatives"
    if re.search(r"share ?-?alike|by-sa", t):
        return None, "share-alike"
    if re.search(r"public domain|\bcc0\b|cc zero", t):
        return "public domain / CC0", None
    if re.search(r"\bcc[ -]?by\b|attribution", t):
        return "CC BY", None
    return None, "no license found"


def lic_line(txt):
    for l in txt.splitlines():
        if re.search(r"licen|cc[ -]?by|cc0|public domain|attribution|share", l, re.I):
            return l.strip()
    return ""


def parse(path):
    """Return {file id: (block text)} using MUSAN's block layout (blocks separated by ==== lines)."""
    out, pending = {}, None
    for b in re.split(r"\n\s*=+\s*\n", open(path, encoding="utf-8", errors="replace").read()):
        ids = ID.findall(b)
        if ids and len(re.findall(r"license:", b, re.I)) > 1:      # one license line per file
            parts = re.split(r"\n(?=(?:music|noise)-[a-z-]+-\d{4}\b)", b)
            for p in parts:
                for i in ID.findall(p)[:1]:
                    out[i] = p
            continue
        has_lic = bool(lic_line(b))
        if ids:
            out.update({i: b if has_lic or not pending else pending + "\n" + b for i in ids})
            pending = None
        elif has_lic:
            pending = b                                            # license printed before its files
    return out


def main():
    keep, rej = [], []
    for sub, typ in [("noise/free-sound", "background"), ("noise/sound-bible", "background"), ("music/fma", "music"),
                     ("music/fma-western-art", "music"), ("music/hd-classical", "music"), ("music/jamendo", "music"), ("music/rfm", "music")]:
        d = M / sub
        files = sorted(d.glob("*.wav"))
        lic_txt = open(d / "LICENSE", encoding="utf-8", errors="replace").read()
        if sub == "noise/free-sound" and "public domain" in lic_txt.lower() and "all selected" in lic_txt.lower():
            blocks = {f.stem: lic_txt.strip() for f in files}
        else:
            blocks = parse(d / "LICENSE")
        for f in files:
            b = blocks.get(f.stem)
            lic, why = classify(b) if b else (None, "file not listed in LICENSE")
            rel = str(f.relative_to(R)).replace("\\", "/")
            if lic:
                split = "eval" if int(hashlib.sha1(f.stem.encode()).hexdigest(), 16) % 5 == 0 else "train"
                keep.append(dict(path=rel, type=typ, split=split, synthetic=False, source=f"MUSAN {sub}", license=lic,
                                 attribution=" | ".join(l.strip() for l in b.splitlines() if l.strip())[:400]))
            else:
                rej.append(dict(path=rel, reason=why, license_text=lic_line(b or "")[:200]))
    out = R / "data/noise"
    json.dump(keep, open(out / "musan_clean.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    json.dump(rej, open(out / "musan_rejected.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    from collections import Counter
    print("kept", Counter((k["source"], k["license"]) for k in keep))
    print("rejected", Counter((r["path"].split("/")[4], r["reason"]) for r in rej))


if __name__ == "__main__":
    main()
