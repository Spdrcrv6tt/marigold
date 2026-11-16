"""Float training of the Marigold intent encoder by distillation to teacher text vectors, plus the CTC value head.

  python sw/intent/train.py --targets data/intent/targets-teacher-v3.npz --tag f1 [--minutes 150]

Data (data/intent/index_train.jsonl, built by data_index.py; never fan or microwave, never eval text):
  TTS training commands and non-commands, LibriSpeech training utterances <= 5.5 s, Speech Commands words.
Augmentation per clip (workers, CPU): optional pre-roll (silence or another short word, 0.1 to 1.0 s, standing in
for the wake phrase and the ring's pre-roll), simple reverb, noise from data/noise train split (fan and microwave
at 0 to 20 dB SNR, babble, TV, music, background, household at 5 to 25 dB), random gain, int16 rounding.
Loss: (1 - cos(q, y)) + in-batch relational KD (KL between teacher-teacher and student-teacher similarity
distributions) + ctc_w * CTC(value tokens).
Writes data/intent/<tag>.pt (state, cfg) and a log line every 200 steps.
"""
import argparse, json, math, os, random, sys, time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sw" / "intent"))
from model import IntentEncoder, CFG, n_weights, enable_qat, load_moonshine  # noqa: E402
import values  # noqa: E402

SR = 16000
MIX = {"tts_cmd": 0.45, "tts_none": 0.10, "libri": 0.30, "sc": 0.15}


def load_index(targets):
    z = np.load(targets, allow_pickle=True)
    tid = {t: i for i, t in enumerate(z["texts"])}
    rows = [json.loads(l) for l in open(ROOT / "data/intent/index_train.jsonl", encoding="utf8")]
    groups = {k: [] for k in MIX}
    for r in rows:
        if r["text"] not in tid:
            continue
        r["tid"] = tid[r["text"]]
        g = ("tts_cmd" if r["kind"] == "cmd" else "tts_none") if r["src"] == "tts" else r["src"]
        groups[g].append(r)
    return groups, z["Y"].astype(np.float32)


class Aug:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        noise = [json.loads(l) for l in open(ROOT / "data/noise/index.jsonl")]
        self.noise = {}
        for r in noise:
            if r["split"] == "train":
                self.noise.setdefault(r["type"], []).append(str(ROOT / r["path"]))
        self.sc = sorted((ROOT / "data/speech_commands").glob("[a-z]*/*.wav"))[::7]

    def read(self, path, n=None):
        info = sf.info(path)
        if n is None or info.frames <= n:
            x, _ = sf.read(path, dtype="float32")
            return x if x.ndim == 1 else x.mean(1)
        s = int(self.rng.integers(0, info.frames - n))
        x, _ = sf.read(path, start=s, stop=s + n, dtype="float32")
        return x if x.ndim == 1 else x.mean(1)

    def __call__(self, x):
        r = self.rng
        x = x / (np.abs(x).max() + 1e-6) * 0.5
        if r.random() < 0.8:                                    # pre-roll 0.1-0.5 s (chip: 0.3 s before the wake sample)
            pre = int(r.uniform(0.1, 0.5) * SR)
            if r.random() < 0.7:                                # tail of a word (stands in for the wake phrase's end)
                w = self.read(str(self.sc[int(r.integers(len(self.sc)))]))
                a = np.abs(w); idx = np.nonzero(a > 0.1 * a.max())[0]
                w = w[: idx[-1] + 1] if len(idx) else w
                tl = min(len(w), int(r.uniform(0.1, 1.0) * pre))
                w = w[len(w) - tl:] / (np.abs(w).max() + 1e-6) * r.uniform(0.2, 0.6)
                x = np.concatenate([w.astype(np.float32), np.zeros(pre - tl, np.float32), x])
            else:
                x = np.concatenate([np.zeros(pre, np.float32), x])
        if r.random() < 0.5:                                    # trailing silence like the 128 ms end-of-speech hang
            x = np.concatenate([x, np.zeros(int(r.uniform(0.05, 0.3) * SR), np.float32)])
        if r.random() < 0.3:                                    # reverb: exponentially decaying noise RIR
            rt = r.uniform(0.15, 0.7); L = int(rt * SR)
            t = np.arange(L) / SR
            h = r.standard_normal(L).astype(np.float32) * np.exp(-6.9 * t / rt).astype(np.float32)
            h[0] = 1.0; h /= np.sqrt((h * h).sum())
            from scipy.signal import fftconvolve
            wet = fftconvolve(x, h)[: len(x)].astype(np.float32)
            x = r.uniform(0.3, 0.8) * x + wet
        if r.random() < 0.75:                                   # noise
            p = r.random()
            if p < 0.4:
                kind = "fan" if r.random() < 0.5 else "microwave"; snr = r.uniform(0, 20)
            else:
                kind = r.choice([k for k in ("babble", "tv", "music", "background", "household", "stationary") if k in self.noise])
                snr = r.uniform(5, 25)
            files = self.noise[kind]
            nz = self.read(files[int(r.integers(len(files)))], len(x))
            if len(nz) < len(x):
                nz = np.tile(nz, len(x) // max(len(nz), 1) + 1)[: len(x)]
            ps = np.mean(x[np.abs(x) > 0.02 * np.abs(x).max()] ** 2) if np.any(x) else 1e-6
            pn = np.mean(nz ** 2) + 1e-9
            x = x + nz * np.sqrt(ps / pn / (10 ** (snr / 10)))
        x = x / (np.abs(x).max() + 1e-6) * (10 ** (r.uniform(-30, -1) / 20))
        return (np.round(x * 32767) / 32768).astype(np.float32)


class Batches(torch.utils.data.IterableDataset):
    def __init__(self, groups, bs, max_s, seed):
        self.groups, self.bs, self.max_s, self.seed = groups, bs, max_s, seed

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        seed = self.seed * 1000 + (wi.id if wi else 0) + int(time.time()) % 1000
        rng = random.Random(seed); aug = Aug(seed)
        keys, w = list(MIX), [MIX[k] for k in MIX]
        while True:
            xs, tids, txt, cm = [], [], [], []
            for g in rng.choices(keys, w, k=self.bs):
                r = rng.choice(self.groups[g])
                try:
                    if "start" in r:
                        x, _ = sf.read(str(ROOT / r["path"]), start=r["start"], stop=r["stop"], dtype="float32")
                    else:
                        x, _ = sf.read(str(ROOT / r["path"]), dtype="float32")
                except Exception:
                    continue
                if x.ndim > 1:
                    x = x.mean(1)
                x = aug(x)[: int(self.max_s * SR)]
                xs.append(x); tids.append(r["tid"]); txt.append(r["text"]); cm.append(0.0 if "start" in r else 1.0)
            n = max(len(x) for x in xs)
            pcm = np.zeros((len(xs), n), np.float32)
            for i, x in enumerate(xs):
                pcm[i, : len(x)] = x
            tg = [values.targets(t) for t in txt]
            tgt = np.zeros((len(tg), max(1, max(len(s) for s in tg))), np.int64)
            for i, s in enumerate(tg):
                tgt[i, : len(s)] = s
            yield (torch.from_numpy(pcm), torch.tensor([len(x) for x in xs]), torch.tensor(tids),
                   torch.from_numpy(tgt), torch.tensor([len(s) for s in tg]), torch.tensor(cm))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--minutes", type=float, default=150)
    ap.add_argument("--steps", type=int, default=40000, help="schedule length (the run also stops at --minutes)")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--max-s", type=float, default=6.0)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--ctc-w", type=float, default=0.3)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--gpu-gb", type=float, default=10)
    ap.add_argument("--init", default="")
    ap.add_argument("--moonshine", action="store_true", help="start the frontend and layers from Moonshine Streaming Tiny")
    ap.add_argument("--lr-pre", type=float, default=0.3, help="LR multiplier for Moonshine-initialised weights")
    ap.add_argument("--qat", action="store_true", help="4/8-bit weight QAT from --init, distilling also from the float model")
    ap.add_argument("--lr-scale", type=float, default=5e-4)
    ap.add_argument("--fw", type=float, default=0.5, help="QAT: weight of 1 - cos(q, float model q)")
    a = ap.parse_args()
    torch.manual_seed(0)
    dev = "cuda"
    torch.cuda.set_per_process_memory_fraction(min(1.0, a.gpu_gb / (torch.cuda.get_device_properties(0).total_memory / 2 ** 30)))
    groups, Y = load_index(a.targets)
    print({k: len(v) for k, v in groups.items()}, "targets", Y.shape, flush=True)
    cfg = dict(CFG); cfg["D"] = Y.shape[1]
    m = IntentEncoder(cfg).to(dev)
    fm = None
    pre = set()
    if a.moonshine and not a.init:
        pre, _ = load_moonshine(m)
        print("moonshine init:", len(pre), "tensors", flush=True)
    if a.init:
        m.load_state_dict(torch.load(a.init, map_location=dev)["state"])
    if a.qat:
        fm = IntentEncoder(cfg).to(dev); fm.load_state_dict(m.state_dict()); fm.eval()
        for p_ in fm.parameters():
            p_.requires_grad_(False)
        enable_qat(m); m.to(dev)
    print("weights", n_weights(m), "qat" if a.qat else "float", flush=True)
    Yt = torch.from_numpy(Y).to(dev)
    dl = torch.utils.data.DataLoader(Batches(groups, a.bs, a.max_s, 1), batch_size=None, num_workers=a.workers,
                                     prefetch_factor=4, persistent_workers=True)
    sc = [p_ for n_, p_ in m.named_parameters() if n_.endswith("log_s")]
    pw = [p_ for n_, p_ in m.named_parameters() if n_ in pre]
    rest = [p_ for n_, p_ in m.named_parameters() if not n_.endswith("log_s") and n_ not in pre]
    groups = [{"params": rest, "lr": a.lr}] + ([{"params": pw, "lr": a.lr * a.lr_pre}] if pw else []) +              ([{"params": sc, "lr": a.lr_scale, "weight_decay": 0.0}] if sc else [])
    opt = torch.optim.AdamW(groups, lr=a.lr, weight_decay=0.01, betas=(0.9, 0.98))
    warm = 200 if a.qat else 1000
    t_start = time.time()
    prog = lambda s: min(1.0, max(s / a.steps, (time.time() - t_start) / (a.minutes * 60)))   # step- or time-based
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * prog(s)))))
    out = ROOT / f"data/intent/{a.tag}.pt"
    t0 = time.time(); stats = []
    step = 0
    it = iter(dl)
    for pcm, lens, tids, tg, tl, cmask in it:
        pcm, lens, tids = pcm.to(dev, non_blocking=True), lens.to(dev), tids.to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            q, logits, flen = m(pcm, lens)
        q = q.float(); y = Yt[tids]; D = q.shape[1]
        cos = (q * y).sum(1) / D
        l_cos = (1 - cos).mean()
        S = q @ y.T / D / a.tau
        Tt = y @ y.T / D / a.tau
        l_kd = F.kl_div(F.log_softmax(S, 1), F.log_softmax(Tt, 1), log_target=True, reduction="batchmean")
        lp = F.log_softmax(logits.float(), -1).transpose(0, 1)
        tl = tl.to(dev); cmask = cmask.to(dev)      # crops of long LibriSpeech utterances: no CTC (estimated words)
        l_ctc = F.ctc_loss(lp, tg.to(dev), flen, tl, blank=0, reduction="none", zero_infinity=True) / tl.clamp_min(1)
        l_ctc = (l_ctc * cmask).sum() / cmask.sum().clamp_min(1)
        loss = l_cos + l_kd + a.ctc_w * l_ctc
        if fm is not None:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                qf, _, _ = fm(pcm, lens)
            loss = loss + a.fw * (1 - (q * qf.float()).sum(1) / D).mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); sched.step(); step += 1
        stats.append([l_cos.item(), l_kd.item(), l_ctc.item(), cos.mean().item()])
        if step % 200 == 0:
            s = np.mean(stats, 0); stats = []
            print(f"step {step} cos_loss {s[0]:.4f} kd {s[1]:.4f} ctc {s[2]:.4f} cos {s[3]:.4f} lr {sched.get_last_lr()[0]:.2e} "
                  f"{(time.time()-t0)/60:.1f} min", flush=True)
        if step % 2000 == 0:
            torch.save({"state": m.state_dict(), "cfg": cfg, "step": step, "qat": a.qat}, out)
        if step >= a.steps or time.time() - t0 > a.minutes * 60:
            break
    torch.save({"state": m.state_dict(), "cfg": cfg, "step": step, "qat": a.qat}, out)
    print("saved", out, "steps", step, flush=True)
    it._shutdown_workers()          # no orphaned workers (they would hold the log pipe open)
    os._exit(0)


if __name__ == "__main__":
    main()
