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


def load_index(targets, protos=None):
    z = np.load(targets, allow_pickle=True)
    tid = {t: i for i, t in enumerate(z["texts"])}
    rows = [json.loads(l) for l in open(ROOT / "data/intent/index_train.jsonl", encoding="utf8")]
    groups = {k: [] for k in MIX}
    pa_id, p_id = {}, {}
    if protos is not None:      # decision labels (iteration 2): (product, action) index, product index, kind
        pa_id = {s_: i for i, s_ in enumerate(protos["pa"])}
        p_id = {s_: i for i, s_ in enumerate(protos["prods"])}
    for r in rows:
        if r["text"] not in tid:
            continue
        r["tid"] = tid[r["text"]]
        g = ("tts_cmd" if r["kind"] == "cmd" else "tts_none") if r["src"] == "tts" else r["src"]
        # dk: 1 = command of a training product (pa >= 0); 0 = non-command speech (TTS 'none' of a known product,
        # LibriSpeech); -1 = no decision loss (Speech Commands words such as 'on', 'off', 'stop' are commands for
        # many products)
        r["pa"] = pa_id.get(f"{r.get('product')}|{r.get('action')}", -1) if g == "tts_cmd" else -1
        r["pp"] = p_id.get(r.get("product"), -1) if r["src"] == "tts" else -1
        r["dk"] = 1 if r["pa"] >= 0 else 0 if (g == "libri" or (g == "tts_none" and r["pp"] >= 0)) else -1
        groups[g].append(r)
    return groups, z["Y"].astype(np.float32)


class Aug:
    P_NOISE, SNR_HI = 0.75, 20.0       # v1; v1.1 passes --noise-p 0.85 --snr-hi 15 (stronger noise)

    def __init__(self, seed, p_noise=0.75, snr_hi=20.0):
        self.P_NOISE, self.SNR_HI = p_noise, snr_hi        # instance values: workers are spawned (Windows)
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
        if r.random() < self.P_NOISE:                           # noise
            p = r.random()
            if p < 0.4:
                kind = "fan" if r.random() < 0.5 else "microwave"; snr = r.uniform(0, self.SNR_HI)
            else:
                kind = r.choice([k for k in ("babble", "tv", "music", "background", "household", "stationary") if k in self.noise])
                snr = r.uniform(5 - (20 - self.SNR_HI), 25 - (20 - self.SNR_HI))
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
    def __init__(self, groups, bs, max_s, seed, prods_per_batch=0, mix=None, p_noise=0.75, snr_hi=20.0):
        self.groups, self.bs, self.max_s, self.seed, self.ppb = groups, bs, max_s, seed, prods_per_batch
        self.mix, self.p_noise, self.snr_hi = dict(mix or MIX), p_noise, snr_hi

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        seed = self.seed * 1000 + (wi.id if wi else 0) + int(time.time()) % 1000
        rng = random.Random(seed); aug = Aug(seed, self.p_noise, self.snr_hi)
        keys, w = list(self.mix), [self.mix[k] for k in self.mix]
        byp = {}
        for r in self.groups["tts_cmd"]:
            byp.setdefault(r.get("product"), []).append(r)
        prods = sorted(byp, key=str)
        while True:
            xs, tids, txt, cm, lab = [], [], [], [], []
            pool = [x for p_ in rng.sample(prods, self.ppb) for x in byp[p_]] if self.ppb else None
            for g in rng.choices(keys, w, k=self.bs):
                # commands of only a few products per batch: the in-batch negatives are the product's other actions
                r = rng.choice(pool) if (g == "tts_cmd" and pool) else rng.choice(self.groups[g])
                try:
                    if r.get("lkd"):              # LibriSpeech crop without text: audio-teacher KD and reject only
                        x = aug.read(str(ROOT / r["path"]), int(rng.uniform(1.5, 5.5) * SR))
                    elif "start" in r:
                        x, _ = sf.read(str(ROOT / r["path"]), start=r["start"], stop=r["stop"], dtype="float32")
                    else:
                        x, _ = sf.read(str(ROOT / r["path"]), dtype="float32")
                except Exception:
                    continue
                if x.ndim > 1:
                    x = x.mean(1)
                if len(x) < 1600 or not np.isfinite(x).all():      # empty or broken clip: skip
                    continue
                x = aug(x)[: int(self.max_s * SR)]
                xs.append(x); tids.append(r["tid"]); txt.append(r["text"]); cm.append(0.0 if ("start" in r or r.get("lkd")) else 1.0)
                lab.append((r.get("pa", -1), r.get("pp", -1), r.get("dk", -1)))
            n = max(len(x) for x in xs)
            pcm = np.zeros((len(xs), n), np.float32)
            for i, x in enumerate(xs):
                pcm[i, : len(x)] = x
            tg = [values.targets(t) for t in txt]
            tgt = np.zeros((len(tg), max(1, max(len(s) for s in tg))), np.int64)
            for i, s in enumerate(tg):
                tgt[i, : len(s)] = s
            yield (torch.from_numpy(pcm), torch.tensor([len(x) for x in xs]), torch.tensor(tids),
                   torch.from_numpy(tgt), torch.tensor([len(s) for s in tg]), torch.tensor(cm), torch.tensor(lab))


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
    ap.add_argument("--ppb", type=int, default=0, help="products per batch for the TTS commands (0 = any)")
    ap.add_argument("--mix", default="", help="sampling weights tts_cmd,tts_none,libri,sc (default 0.45,0.10,0.30,0.15)")
    ap.add_argument("--noise-p", type=float, default=0.75)
    ap.add_argument("--snr-hi", type=float, default=20.0)
    ap.add_argument("--moonshine", action="store_true", help="start the frontend and layers from Moonshine Streaming Tiny")
    ap.add_argument("--lr-pre", type=float, default=0.3, help="LR multiplier for Moonshine-initialised weights")
    ap.add_argument("--qat", action="store_true", help="4/8-bit weight QAT from --init, distilling also from the float model")
    ap.add_argument("--lr-scale", type=float, default=5e-4)
    ap.add_argument("--float-ref", default="", help="QAT: the float model to distill from (default --init)")
    ap.add_argument("--fw", type=float, default=0.5, help="QAT: weight of 1 - cos(q, float model q)")
    ap.add_argument("--protos", default="", help="decision loss (iteration 2): data/intent/protos-<teacher>.npz")
    ap.add_argument("--dec-w", type=float, default=1.0, help="weight of the decision loss")
    ap.add_argument("--dec-tau", type=float, default=0.05, help="decision softmax temperature (cosine units)")
    ap.add_argument("--dec-m", type=float, default=0.05, help="additive margin on the true class (cosine units)")
    ap.add_argument("--aux-w", type=float, default=1.0, help="weight of the distillation losses (1 - cos, relational KD)")
    ap.add_argument("--thr0", type=float, default=0.45, help="initial reject threshold of the decision loss (cosine)")
    ap.add_argument("--nl", type=int, default=0, help="encoder layers (0 = CFG's 2); 6 = the full Moonshine encoder (PC-side audio teacher)")
    ap.add_argument("--fut", default="", help="look-ahead frames per layer, comma-separated (Moonshine's own: 4,4,0,0,4,4)")
    ap.add_argument("--teacher-audio", default="", help="frozen audio-teacher checkpoint (same q space) for audio-to-audio KD")
    ap.add_argument("--akd-w", type=float, default=1.0, help="weight of the audio-teacher KD (1 - cos plus relational KD)")
    ap.add_argument("--lkd", type=float, default=0.0, help="sampling weight of LibriSpeech crops without text (all train subsets)")
    ap.add_argument("--min-spm", type=float, default=0.0, help="exit code 3 if fewer steps per minute after 2 minutes")
    a = ap.parse_args()
    torch.manual_seed(0)
    if a.mix:
        for k_, v_ in zip(list(MIX), a.mix.split(",")):
            MIX[k_] = float(v_)
    dev = "cuda"
    torch.cuda.set_per_process_memory_fraction(min(1.0, a.gpu_gb / (torch.cuda.get_device_properties(0).total_memory / 2 ** 30)))
    PZ = np.load(a.protos, allow_pickle=True) if a.protos else None
    groups, Y = load_index(a.targets, PZ)
    print({k: len(v) for k, v in groups.items()}, "targets", Y.shape, flush=True)
    cfg = dict(CFG); cfg["D"] = Y.shape[1]
    if a.nl:
        cfg["NL"] = a.nl
    if a.fut:
        cfg["FUT"] = tuple(int(v_) for v_ in a.fut.split(","))
    if a.lkd > 0:          # every LibriSpeech training utterance (100 + 360 + 500 h when present), random crops, no text
        MIX["lkd"] = a.lkd
        groups["lkd"] = [{"path": str(f.relative_to(ROOT)).replace("\\", "/"), "text": "", "tid": -1, "lkd": True, "pa": -1, "pp": -1, "dk": 0}
                         for base in ("data/librispeech/LibriSpeech/train-clean-100", "data/librispeech-360/LibriSpeech/train-clean-360",
                                      "data/librispeech-extra/LibriSpeech/train-other-500") if (ROOT / base).is_dir()
                         for f in sorted((ROOT / base).rglob("*.flac"))]
        print("lkd utterances", len(groups["lkd"]), flush=True)
    ta = None
    if a.teacher_audio:
        ckt = torch.load(ROOT / a.teacher_audio, map_location=dev)
        ta = IntentEncoder(ckt["cfg"]).to(dev); ta.load_state_dict(ckt["state"]); ta.eval()
        for p_ in ta.parameters():
            p_.requires_grad_(False)
        print("audio teacher", a.teacher_audio, "layers", ckt["cfg"]["NL"], flush=True)
    m = IntentEncoder(cfg).to(dev)
    fm = None
    pre = set()
    if a.moonshine and not a.init:
        pre, _ = load_moonshine(m)
        print("moonshine init:", len(pre), "tensors", flush=True)
    ck0 = torch.load(a.init, map_location=dev) if a.init else None
    if ck0 is not None and ck0.get("qat"):          # resuming a QAT run
        enable_qat(m); m.to(dev); m.load_state_dict(ck0["state"])
    elif ck0 is not None:
        m.load_state_dict(ck0["state"])
    if a.qat:
        fm = IntentEncoder(cfg).to(dev)
        fm.load_state_dict(torch.load(a.float_ref or a.init, map_location=dev)["state"]); fm.eval()
        for p_ in fm.parameters():
            p_.requires_grad_(False)
        if not (ck0 is not None and ck0.get("qat")):
            enable_qat(m); m.to(dev)
    print("weights", n_weights(m), "qat" if a.qat else "float", flush=True)
    Yt = torch.from_numpy(Y).to(dev)
    if PZ is not None:   # fixed teacher prototypes, unit L2; reject threshold learned (a scalar, cosine units)
        PA = F.normalize(torch.from_numpy(PZ["A"]).float().to(dev), dim=-1)          # [n_pa, K, D]
        PR = F.normalize(torch.from_numpy(PZ["R"]).float().to(dev), dim=-1)          # [n_p, Kr, D]
        pa_prod = torch.from_numpy(PZ["pa_prod"].astype(np.int64)).to(dev)          # [n_pa]
        thr_p = torch.nn.Parameter(torch.tensor(float(a.thr0), device=dev))
        print("decision prototypes", tuple(PA.shape), tuple(PR.shape), "rows with labels",
              {k: sum(r["dk"] >= 0 for r in v) for k, v in groups.items()}, flush=True)
    dl = torch.utils.data.DataLoader(Batches(groups, a.bs, a.max_s, 1, a.ppb, MIX, a.noise_p, a.snr_hi), batch_size=None, num_workers=a.workers,
                                     prefetch_factor=4, persistent_workers=True)
    sc = [p_ for n_, p_ in m.named_parameters() if n_.endswith("log_s")]
    pw = [p_ for n_, p_ in m.named_parameters() if n_ in pre]
    rest = [p_ for n_, p_ in m.named_parameters() if not n_.endswith("log_s") and n_ not in pre]
    groups = [{"params": rest, "lr": a.lr}] + ([{"params": pw, "lr": a.lr * a.lr_pre}] if pw else []) + \
             ([{"params": [thr_p], "lr": 1e-3, "weight_decay": 0.0}] if PZ is not None else []) +              ([{"params": sc, "lr": a.lr_scale, "weight_decay": 0.0}] if sc else [])
    opt = torch.optim.AdamW(groups, lr=a.lr, weight_decay=0.01, betas=(0.9, 0.98))
    warm = 200 if a.qat else 1000
    t_start = time.time()
    prog = lambda s: min(1.0, max(s / a.steps, (time.time() - t_start) / (a.minutes * 60)))   # step- or time-based
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * prog(s)))))
    out = ROOT / f"data/intent/{a.tag}.pt"
    t0 = time.time(); stats = []
    step = 0
    it = iter(dl)
    for pcm, lens, tids, tg, tl, cmask, lab in it:
        pcm, lens, tids = pcm.to(dev, non_blocking=True), lens.to(dev), tids.to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            q, logits, flen = m(pcm, lens)
        q = q.float(); tm = tids >= 0; y = Yt[tids.clamp_min(0)]; D = q.shape[1]
        cos = (q * y).sum(1) / D
        l_cos = ((1 - cos) * tm).sum() / tm.sum().clamp_min(1)
        S = q[tm] @ y[tm].T / D / a.tau
        Tt = y[tm] @ y[tm].T / D / a.tau
        l_kd = F.kl_div(F.log_softmax(S, 1), F.log_softmax(Tt, 1), log_target=True, reduction="batchmean")
        cos = cos[tm]
        lp = F.log_softmax(logits.float(), -1).transpose(0, 1)
        tl = tl.to(dev); cmask = cmask.to(dev)      # crops of long LibriSpeech utterances: no CTC (estimated words)
        cmask = cmask * (tl <= flen).float()        # a target longer than the frames cannot be aligned
        tl = torch.minimum(tl, flen)
        l_ctc = F.ctc_loss(lp, tg.to(dev), flen, tl, blank=0, reduction="none", zero_infinity=True) / tl.clamp_min(1)
        l_ctc = (l_ctc * cmask).sum() / cmask.sum().clamp_min(1)
        loss = a.aux_w * (l_cos + l_kd) + a.ctc_w * l_ctc
        l_dec = torch.zeros((), device=dev); dacc = float("nan")
        if PZ is not None:   # the chip's decision as a softmax: actions of the product (max over prototypes) + reject
            lab = lab.to(dev); pa, pp, dk = lab[:, 0], lab[:, 1], lab[:, 2]
            qn = F.normalize(q, dim=1)
            SA = torch.einsum("bd,pkd->bpk", qn, PA).amax(-1)                       # [B, n_pa]
            SR = torch.einsum("bd,pkd->bpk", qn, PR).amax(-1)                       # [B, n_p]
            thr = thr_p.clamp(0.0, 0.9)
            # pairs (sample, product): commands and TTS 'none' against their own product; LibriSpeech against every
            # training product's actions at once (reject = the best of all reject prototypes or the threshold)
            own = (dk >= 0) & (pp >= 0)
            lib = (dk == 0) & (pp < 0)
            rows_ = []
            if own.any():
                i = own.nonzero().squeeze(1)
                L = SA[i].masked_fill(pa_prod[None, :] != pp[i][:, None], -4.0)
                rj = torch.maximum(SR[i, pp[i]], thr)
                tg_ = torch.where(dk[i] == 1, pa[i], torch.full_like(pa[i], SA.shape[1]))
                rows_.append((torch.cat([L, rj[:, None]], 1), tg_))
            if lib.any():
                i = lib.nonzero().squeeze(1)
                rj = torch.maximum(SR[i].amax(1), thr)
                rows_.append((torch.cat([SA[i], rj[:, None]], 1), torch.full_like(i, SA.shape[1])))
            if rows_:
                Lg = torch.cat([r_[0] for r_ in rows_]); tg_ = torch.cat([r_[1] for r_ in rows_])
                Lm = Lg - a.dec_m * F.one_hot(tg_, Lg.shape[1]).float()
                l_dec = F.cross_entropy(Lm / a.dec_tau, tg_)
                dacc = float((Lg.argmax(1) == tg_).float().mean())
                loss = loss + a.dec_w * l_dec
        l_akd = torch.zeros((), device=dev)
        if ta is not None:     # audio-to-audio KD from the frozen audio teacher on the same augmented audio
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                qt, _, _ = ta(pcm, lens)
            qt = qt.float()
            l_akd = (1 - (q * qt).sum(1) / D).mean() + F.kl_div(F.log_softmax(q @ qt.T / D / a.tau, 1),
                                                                 F.log_softmax(qt @ qt.T / D / a.tau, 1), log_target=True, reduction="batchmean")
            loss = loss + a.akd_w * l_akd
        if fm is not None:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                qf, _, _ = fm(pcm, lens)
            loss = loss + a.fw * (1 - (q * qf.float()).sum(1) / D).mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); sched.step(); step += 1
        if a.min_spm and not getattr(main, "_spm", False) and time.time() - t0 >= 120:
            main._spm = True
            spm = step / ((time.time() - t0) / 60)
            print(f"throughput: {step} steps in {time.time() - t0:.0f} s = {spm:.0f} steps/min; GPU max reserved "
                  f"{torch.cuda.max_memory_reserved() / 2 ** 30:.1f} GB", flush=True)
            if spm < a.min_spm:
                print(f"ABORT: below --min-spm {a.min_spm}", flush=True)
                it._shutdown_workers(); os._exit(3)
        stats.append([l_cos.item(), l_kd.item(), l_ctc.item(), cos.mean().item(), l_dec.item(), dacc, l_akd.item()])
        if step % 200 == 0:
            s = np.nanmean(stats, 0); stats = []
            print(f"step {step} cos_loss {s[0]:.4f} kd {s[1]:.4f} ctc {s[2]:.4f} cos {s[3]:.4f} dec {s[4]:.4f} dacc {s[5]:.3f} akd {s[6]:.4f} "
                  f"thr {float(thr_p) if PZ is not None else 0:.3f} lr {sched.get_last_lr()[0]:.2e} {(time.time()-t0)/60:.1f} min", flush=True)
        if step % 1000 == 0:
            torch.save({"state": m.state_dict(), "cfg": cfg, "step": step, "qat": a.qat,
                        "thr": float(thr_p) if PZ is not None else None}, out)
        if step >= a.steps or time.time() - t0 > a.minutes * 60:
            break
    torch.save({"state": m.state_dict(), "cfg": cfg, "step": step, "qat": a.qat}, out)
    print("saved", out, "steps", step, flush=True)
    it._shutdown_workers()          # no orphaned workers (they would hold the log pipe open)
    os._exit(0)


if __name__ == "__main__":
    main()
