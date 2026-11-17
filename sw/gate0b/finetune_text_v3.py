"""Gate 0b: fine-tune a small text embedder so command phrasings cluster by action and
non-commands stay away from every action. Trained only on the training products
(data/gate0b/train_text.json, no fans or microwaves); fixed recipe, no tuning on eval/.

Loss: supervised contrastive within one product per batch (actions of other products
never act as negatives, so "turn on" for a lamp and for a heater are not pushed apart).
Non-commands are negatives only (no positives of their own).

v3 (teacher): --data picks the training text (default train_text_v3.json), --holdout leaves products out
for model selection, and every line (commands, non-commands, LibriSpeech snippets) passes the eval
leak filter (sw/textfilter.py).

Usage: python sw/gate0b/finetune_text_v3.py BASE OUTDIR [--steps 2000] [--holdout "a,b"] [--tau 0.1 --libri 32]
"""
import argparse, json, os, random
from pathlib import Path

import sys
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from textfilter import EvalFilter  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(ROOT / "data" / "hf"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base"); ap.add_argument("out")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--per_action", type=int, default=6)
    ap.add_argument("--none", type=int, default=24)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default="data/gate0b/train_text_v3.json")
    ap.add_argument("--holdout", default="", help="comma-separated products left out (model selection)")
    ap.add_argument("--libri", type=int, default=0, help="LibriSpeech transcript snippets per batch, each its own class (SimCSE pair)")
    a = ap.parse_args()
    random.seed(a.seed); torch.manual_seed(a.seed)
    from sentence_transformers import SentenceTransformer
    flt = EvalFilter()
    hold = {h.strip() for h in a.holdout.split(",") if h.strip()}
    data = json.load(open(ROOT / a.data, encoding="utf8"))
    data = {p: {"actions": {k: flt.keep(v) for k, v in r["actions"].items()}, "none": flt.keep(r["none"])}
            for p, r in data.items() if p not in hold}
    print("products", len(data), "held out", sorted(hold), "eval-identical lines dropped", flt.dropped, flush=True)
    # Generic non-commands shared by all products (pooled from every product's 'none').
    pool_none = sorted({t for r in data.values() for t in r["none"]})
    libri = []
    if a.libri:  # LibriSpeech training transcripts (CC BY 4.0), cut into 2 to 10 word spans
        for f in sorted((ROOT / "data/librispeech").rglob("*.trans.txt")) + sorted((ROOT / "data/librispeech-extra").rglob("*.trans.txt")):
            if "test" in str(f) or "dev" in str(f):
                continue
            for line in open(f, encoding="utf8"):
                w = line.split()[1:]
                if len(w) >= 2:
                    n = random.randint(2, min(10, len(w))); s = random.randint(0, len(w) - n)
                    t = " ".join(w[s:s + n]).lower()
                    if flt.ok(t):
                        libri.append(t)
        print("libri snippets", len(libri), flush=True)
    prods = [p for p in data if len(data[p]["actions"]) >= 2]
    m = SentenceTransformer(a.base, device="cuda")
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 100) * max(0.0, 1 - s / a.steps))
    for step in range(a.steps):
        p = random.choice(prods)
        texts, labels = [], []
        for li, (act, phr) in enumerate(data[p]["actions"].items()):
            if len(phr) < 2:
                continue
            for t in random.sample(phr, min(a.per_action, len(phr))):
                texts.append(t); labels.append(li)
        nn = random.sample(data[p]["none"], min(a.none // 2, len(data[p]["none"]))) + random.sample(pool_none, a.none // 2)
        texts += nn; labels += [-1] * len(nn)
        if libri:  # each snippet twice; dropout makes the two views differ; own class keeps the space spread
            for j, t in enumerate(random.sample(libri, a.libri)):
                texts += [t, t]; labels += [1000 + j, 1000 + j]
        feats = m.tokenize(texts)
        feats = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in feats.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            e = m(feats)["sentence_embedding"]
        e = torch.nn.functional.normalize(e.float(), dim=-1)
        y = torch.tensor(labels, device="cuda")
        sim = e @ e.T / a.tau
        eye = torch.eye(len(y), dtype=torch.bool, device="cuda")
        sim = sim.masked_fill(eye, -1e9)
        pos = (y[:, None] == y[None, :]) & ~eye & (y[:, None] >= 0)
        logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        anchors = pos.any(1)
        loss = -(logp * pos).sum(1)[anchors] / pos.sum(1)[anchors]
        loss = loss.mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 100 == 0 or step == a.steps - 1:
            print(f"step {step} loss {loss.item():.4f}", flush=True)
    m.eval()
    m.save(a.out)
    print("saved", a.out, flush=True)


if __name__ == "__main__":
    main()
