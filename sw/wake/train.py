"""Train the wake phone model with CTC on exact integer front-end features computed on the GPU.

  python -I sw/wake/train.py float  OUT_DIR [--epochs N]           # float model with BatchNorm
  python -I sw/wake/train.py qat    OUT_DIR --init CKPT [--epochs N] # BN folded, 4-bit weights, uint8 activations

Augmentation on the GPU, in the waveform domain, before the integer front end: gain -30..+3 dB, train-split
noise (data/noise, split=train) at 0..25 dB SNR for half the utterances, synthetic room reverb for a quarter.
"""
import argparse, json, math, random, sys, time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fe
from model import Net, n_weights
from lexicon import R

DEV = "cuda"
MAX_N = 17 * 16000


class DS(torch.utils.data.Dataset):
    def __init__(self, rows):
        self.paths = [r["path"] for r in rows]                 # compact: workers get a pickled copy
        self.ph = [np.asarray(r["ph"], np.int8) for r in rows]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        x, sr = sf.read(self.paths[i], dtype="int16")
        if x.ndim > 1:
            x = x[:, 0]
        return torch.from_numpy(x[:MAX_N].copy()), torch.from_numpy(self.ph[i].astype(np.int64))


def batches(rows, budget, seed):
    idx = sorted(range(len(rows)), key=lambda i: rows[i]["n"])
    out, cur, mx = [], [], 0
    for i in idx:
        n = min(rows[i]["n"], MAX_N)
        if cur and max(mx, n) * (len(cur) + 1) > budget:
            out.append(cur); cur, mx = [], 0
        cur.append(i); mx = max(mx, n)
    if cur:
        out.append(cur)
    random.Random(seed).shuffle(out)
    return out


def collate(items):
    n = max(max(len(x) for x, _ in items), 560 + 320)
    X = torch.zeros(len(items), n, dtype=torch.int16)
    lens = []
    for k, (x, _) in enumerate(items):
        X[k, :len(x)] = x
        lens.append(len(x))
    ph = [p for _, p in items]
    return X, torch.tensor(lens), torch.cat(ph), torch.tensor([len(p) for p in ph])


def load_noise(max_h=2.5, seed=0):
    rows = [json.loads(l) for l in open(R / "data/noise/index.jsonl", encoding="utf-8")]
    rows = [r for r in rows if r["split"] == "train"]
    random.Random(seed).shuffle(rows)
    bufs, tot = [], 0
    for r in rows:
        x, sr = sf.read(str(R / r["path"]), dtype="float32")
        if x.ndim > 1:
            x = x.mean(1)
        if sr != 16000:
            continue
        x = x / (np.sqrt(np.mean(x ** 2)) + 1e-8)          # unit power per file
        bufs.append(x.astype(np.float16)); tot += len(x)
        if tot > max_h * 3600 * 16000:
            break
    print(f"noise pool {tot / 16000 / 3600:.2f} h from {len(bufs)} files", flush=True)
    return torch.from_numpy(np.concatenate(bufs)).to(DEV)


AUG = dict(noise_p=0.5, snr_lo=0.0, snr_hi=25.0, rev_p=0.25)


def augment(X, lens, noise, g):
    """X int16 (B, n) on GPU -> int64 samples after reverb, noise and gain."""
    B, n = X.shape
    x = X.float()
    mask = (torch.arange(n, device=DEV)[None] < lens[:, None].to(DEV)).float()
    # reverb on a quarter
    rv = torch.rand(B, generator=g, device=DEV) < AUG["rev_p"]
    if rv.any():
        L = 8000
        t = torch.arange(L, device=DEV) / 16000.0
        rt60 = 0.2 + 0.6 * torch.rand(B, 1, generator=g, device=DEV)
        tail = torch.randn(B, L, generator=g, device=DEV) * torch.exp(-6.9 * t[None] / rt60)
        tail[:, :16] = 0
        drr = 10 ** ((-3 + 13 * torch.rand(B, 1, generator=g, device=DEV)) / 10)     # direct / reverberant energy
        tail = tail / torch.sqrt((tail ** 2).sum(1, keepdim=True) * drr)
        h = tail; h[:, 0] = 1.0
        nf = 1 << int(math.ceil(math.log2(n + L)))
        y = torch.fft.irfft(torch.fft.rfft(x, nf) * torch.fft.rfft(h, nf), nf)[:, :n]
        x = torch.where(rv[:, None], y, x)
    # noise on half
    nz = torch.rand(B, generator=g, device=DEV) < AUG["noise_p"]
    if nz.any():
        p_s = (x ** 2 * mask).sum(1) / mask.sum(1).clamp(min=1)
        snr = AUG["snr_lo"] + (AUG["snr_hi"] - AUG["snr_lo"]) * torch.rand(B, generator=g, device=DEV)
        off = torch.randint(0, len(noise) - n, (B,), generator=g, device=DEV)
        seg = noise[off[:, None] + torch.arange(n, device=DEV)[None]].float()
        scale = torch.sqrt(p_s / 10 ** (snr / 10))
        x = x + torch.where(nz, scale, torch.zeros_like(scale))[:, None] * seg * mask
    gain = 10 ** ((-30 + 33 * torch.rand(B, 1, generator=g, device=DEV)) / 20)
    x = torch.round(x * gain).clamp(-32768, 32767) * mask
    return x.long()


def per(model, dev_rows, n=300):
    """Greedy CTC phone error rate on clean dev-clean utterances."""
    model.eval()
    errs = tot = 0
    with torch.no_grad():
        for r in dev_rows[:n]:
            x, _ = sf.read(r["path"], dtype="int16")
            u = fe.features(torch.from_numpy(x.astype(np.int64)).to(DEV))
            lp = model(u[None])[0].argmax(-1).tolist()
            hyp, prev = [], 0
            for c in lp:
                if c != prev and c != 0:
                    hyp.append(c)
                prev = c
            ref = r["ph"]
            d = list(range(len(hyp) + 1))
            for i in range(1, len(ref) + 1):
                pd, d[0] = d[0], i
                for j in range(1, len(hyp) + 1):
                    pd, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, pd + (ref[i - 1] != hyp[j - 1]))
            errs += d[-1]; tot += len(ref)
    model.train()
    return errs / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["float", "qat"])
    ap.add_argument("out")
    ap.add_argument("--epochs", type=float, default=6)
    ap.add_argument("--init", default=None)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--budget", type=int, default=400 * 16000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--aug", default=None, help="noise_p,snr_lo,snr_hi,rev_p")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.aug:
        AUG.update(zip(("noise_p", "snr_lo", "snr_hi", "rev_p"), map(float, a.aug.split(","))))
    print("augmentation", AUG, flush=True)
    torch.manual_seed(0)
    rows = [json.loads(l) for l in open(R / "data/wake/train.jsonl", encoding="utf-8")]
    dev_rows = [json.loads(l) for l in open(R / "data/wake/dev.jsonl", encoding="utf-8")]
    random.Random(1).shuffle(dev_rows)
    noise = load_noise()
    if a.mode == "float":
        model = Net().to(DEV)
    else:
        from qat import QNet
        model = QNet(a.init).to(DEV)
        model.calibrate(rows, DEV)
    print("weights", n_weights(model) if a.mode == "float" else "qat", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4 if a.mode == "float" else 0.0)
    nb = len(batches(rows, a.budget, 0))
    total = int(a.epochs * nb)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total, pct_start=0.08 if a.mode == "float" else 0.02)
    g = torch.Generator(device=DEV); g.manual_seed(1234)
    step, t0, ep = 0, time.time(), 0
    best = 9.9
    skip = 0
    if a.resume and (out / "state.pt").exists():
        st = torch.load(out / "state.pt", map_location=DEV, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        step, ep, best = st["step"], st["ep"], st["best"]; skip = st["in_ep"]
        g.set_state(st["g"].cpu() if hasattr(st["g"], "cpu") else st["g"])
        print(f"resumed at step {step} epoch {ep} (+{skip} batches)", flush=True)
    ds = DS(rows)
    while step < total:
        bl = batches(rows, a.budget, ep)[skip:]
        dl = torch.utils.data.DataLoader(ds, batch_sampler=bl, collate_fn=collate,
                                         num_workers=a.workers, persistent_workers=False, prefetch_factor=2)
        in_ep = skip
        skip = 0
        for X, lens, tg, tl in dl:
            in_ep += 1
            X, lens = X.to(DEV, non_blocking=True), lens.to(DEV)
            with torch.no_grad():
                x = augment(X, lens, noise, g)
                u = fe.features(x)                                   # (B, P, 64) exact integer codes
            P = u.shape[1]
            il = ((lens - 560) // 320 + 1).clamp(min=1, max=P)
            logits = model(u)
            lp = F.log_softmax(logits.float(), -1).transpose(0, 1)
            loss = F.ctc_loss(lp, tg.to(DEV), il, tl.to(DEV), blank=0, reduction="mean", zero_infinity=True)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step(); step += 1
            if step % 100 == 0:
                print(f"step {step}/{total} ep {ep} loss {loss.item():.3f} lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s "
                      f"mem {torch.cuda.max_memory_allocated() / 2**30:.2f}G", flush=True)
            if step % 500 == 0:
                torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), step=step, ep=ep,
                                best=best, in_ep=in_ep, g=g.get_state()), out / "state.pt")
            if step % 1000 == 0 or step == total:
                p = per(model, dev_rows)
                print(f"== step {step} dev-clean PER {p:.4f}", flush=True)
                torch.save(model.state_dict(), out / "last.pt")
                if p < best:
                    best = p
                    torch.save(model.state_dict(), out / "best.pt")
            if step >= total:
                break
        ep += 1
    print(f"done best PER {best:.4f} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
