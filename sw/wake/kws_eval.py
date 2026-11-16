"""Wake evaluation: model -> logits z (1/16 nat, int16) on cached features, batched exact keyword DP on the GPU
(same integer rules as golden.kws, run per stream chunk of at most CHUNK frames), detection rate per
condition and false wakes per 24 h per source, over a sweep of wake_sens.

  kws_eval.py MODEL OUT.json [--clip 96] [--split test|dev]
MODEL: float checkpoint (Net), qat checkpoint (QNet, fake-quant) or int:<wake_int.npz> (integer golden net).
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lexicon import R
import template as TPL

E = R / "data/wake/eval"
DEV = "cuda"
NEG = -32768
CHUNK = 15000                      # 5 min of 20 ms frames
SENS = list(range(0, 41))
PHRASES = ["hey breeze", "hey kitchen", "hey marigold", "okay toaster", "hello lumen", "hey computer"]
FA_SPLIT = {"dev": ["libri_dev_clean", "musan_speech_usgov", "musan_music_a"],
            "test": ["libri_test_clean", "libri_test_other", "musan_speech_librivox", "musan_music_b", "tv_mix"]}


def runner(spec):
    if spec.startswith("int:"):
        sys.path.insert(0, str(Path(__file__).resolve().parent / "golden"))
        import golden
        P = golden.load_params(spec[4:])
        return lambda U: int_net_torch(P, U)
    sd = torch.load(spec, map_location="cpu", weights_only=False)
    if any(k.startswith("Ws.") for k in sd):
        from qat import QNet
        m = QNet(spec).to(DEV).eval()
    else:
        from model import Net
        m = Net().to(DEV).eval(); m.load_state_dict(sd)

    @torch.no_grad()
    def f(U):
        out = []
        for i in range(0, len(U), 200000):          # causal model: chunk with 64 frames of left context
            lo = max(0, i - 64)
            u = torch.from_numpy(U[lo:i + 200000]).to(DEV)
            z = m(u[None])[0][i - lo:]
            out.append(torch.round(z.float() * 16).clamp(-32768, 32767).short().cpu().numpy())
        return np.concatenate(out)
    return f


def int_net_torch(P, U):
    """golden.net in float64 on the GPU (exact: all intermediate integers < 2^53)."""
    out = []
    for i in range(0, len(U), 200000):
        lo = max(0, i - 64)
        x = torch.from_numpy(U[lo:i + 200000].astype(np.float64)).to(DEV) - 128
        T = len(x)
        for l in P:
            W = torch.from_numpy(l["W"].astype(np.float64)).to(DEV)
            b = torch.from_numpy(l["b"].astype(np.float64)).to(DEV)
            if l["kind"] == "dw":
                d = l["dil"]
                xp = torch.cat([torch.zeros((4 * d, x.shape[1]), dtype=x.dtype, device=DEV), x])
                acc = b[None].repeat(T, 1)
                for k in range(5):
                    acc = acc + W[None, :, 0, k] * xp[k * d:k * d + T]
            else:
                acc = x @ W[:, :, 0].T + b[None]
            M = torch.from_numpy(l["M"].astype(np.float64)).to(DEV)
            y = torch.floor((acc * M[None] + 2.0 ** (l["S"] - 1)) / 2.0 ** l["S"])
            if l["kind"] == "out":
                x = y.clamp(-32768, 32767); break
            x = y.clamp(0, 255)
        out.append(x[i - lo:].short().cpu().numpy())
    return np.concatenate(out)


def build_states(tpls):
    """All variants of all templates as one state vector. Returns dict of index tensors."""
    lab, src1, src2, fresh, fin = [], [], [], [], []
    owner, K = [], []
    for ti, tpl in enumerate(tpls):
        var, clip, refr = TPL.parse(tpl)["variants"], tpl[2], tpl[3]
        for v in var:
            base = len(lab)
            n = 2 * len(v) + 1
            for s in range(n):
                lab.append(0 if s % 2 == 0 else v[s // 2])
                src1.append(base + s - 1 if s >= 1 else -1)
                src2.append(base + s - 2 if (s % 2 == 1 and s >= 3 and v[s // 2] != v[s // 2 - 1]) else -1)
                fresh.append(s <= 1)
            fin.append((base + n - 2, base + n - 1, ti, len(v)))
    t = lambda a, dt=torch.long: torch.tensor(a, dtype=dt, device=DEV)
    S = len(lab)
    return dict(lab=t(lab), src1=t([s if s >= 0 else S for s in src1]), src2=t([s if s >= 0 else S for s in src2]),
                fresh=t(fresh, torch.bool), fa=t([f[0] for f in fin]), fb=t([f[1] for f in fin]),
                fowner=t([f[2] for f in fin]), fK=t([f[3] for f in fin]), S=S, nt=len(tpls))


def dp_batch(Z, lens, tpls, sens_list, clip, refr_n):
    """Z: (B, T, 40) int16 numpy (padded), lens (B,). Exact golden semantics per (stream, template, sens).
    Returns fires: list over templates of (n_sens, B) counts, and first-fire frame arrays (n_sens, B, nt) -1 if none
    plus all fire frames as a list of (b, sens_idx, tpl, t)."""
    st = build_states(tpls)
    B, T, _ = Z.shape
    ns, nt = len(sens_list), st["nt"]
    Zt = torch.from_numpy(Z).to(DEV)
    lens_t = torch.from_numpy(np.asarray(lens)).to(DEV)
    A = torch.full((ns, B, st["S"] + 1), NEG, dtype=torch.int32, device=DEV)      # last column: NEG sentinel
    refr = torch.zeros((ns, B, nt), dtype=torch.int32, device=DEV)
    sens = torch.tensor(sens_list, dtype=torch.int32, device=DEV)[:, None, None]
    count = torch.zeros((ns, B, nt), dtype=torch.int32, device=DEV)
    events = []
    owner_state = torch.zeros(st["S"] + 1, dtype=torch.long, device=DEV)
    for i in range(len(st["fa"])):
        pass
    # map every state to its template for resets
    lab_owner = []
    for ti, tpl in enumerate(tpls):
        n = sum(2 * len(v) + 1 for v in TPL.parse(tpl)["variants"])
        lab_owner += [ti] * n
    lab_owner = torch.tensor(lab_owner + [nt], device=DEV)
    for t in range(T):
        z = Zt[:, t].int()                                                    # (B, 40)
        d = torch.clamp(z - z.max(1, keepdim=True).values, min=-clip)         # (B, 40)
        e = d[:, st["lab"]]                                                   # (B, S)
        prev = A
        c = prev[..., :-1].clone()
        c = torch.maximum(c, prev[..., st["src1"]])
        c = torch.maximum(c, prev[..., st["src2"]])
        c[..., st["fresh"]] = torch.clamp(c[..., st["fresh"]], min=0)
        new = torch.clamp(c + e[None], min=NEG)
        sc = torch.maximum(new[..., st["fa"]], new[..., st["fb"]]) + st["fK"].int() * sens[..., 0][..., None]   # (ns, B, nvar)
        best = torch.full((ns, B, nt), -2**30, dtype=torch.int32, device=DEV)
        best = best.scatter_reduce(2, st["fowner"][None, None].expand(ns, B, -1), sc, reduce="amax")
        alive = (t < lens_t)[None, :, None]
        can = (refr == 0) & alive
        fire = can & (best >= 0)
        refr = torch.where(alive & (refr > 0), refr - 1, refr)
        refr = torch.where(fire, torch.full_like(refr, refr_n), refr)
        count += fire.int()
        if fire.any():
            idx = fire.nonzero().tolist()
            events += [(b, s_i, ti, t) for s_i, b, ti in idx]
            # reset the states of fired templates
            fm = torch.cat([fire, torch.zeros_like(fire[..., :1])], -1)[..., lab_owner]   # (ns, B, S+1)
            new = torch.where(fm[..., :-1], torch.full_like(new, NEG), new)
        A = torch.cat([new, prev[..., -1:]], -1)
    return count.cpu().numpy(), events


def chunks(u, off, chunk=CHUNK):
    out = []
    for a, b in zip(off[:-1], off[1:]):
        for s in range(a, b, chunk):
            out.append((s, min(b, s + chunk)))
    return out


def run_streams(zall, spans, tpls, sens, clip, refr, bsz=1024):
    counts, events = None, []
    for i in range(0, len(spans), bsz):
        sp = spans[i:i + bsz]
        T = max(b - a for a, b in sp)
        Z = np.zeros((len(sp), T, 40), np.int16)
        for k, (a, b) in enumerate(sp):
            Z[k, :b - a] = zall[a:b]
        c, ev = dp_batch(Z, [b - a for a, b in sp], tpls, sens, clip, refr)
        counts = c if counts is None else np.concatenate([counts, c], 1)
        events += [(b + i, s, ti, t) for b, s, ti, t in ev]
    return counts, events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--clip", type=int, default=TPL.CLIP)
    ap.add_argument("--refr", type=int, default=TPL.REFRACTORY)
    ap.add_argument("--split", default="test")
    ap.add_argument("--skip-fa", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    f = runner(a.model)
    tpls = []
    for p in PHRASES:
        tp = bytearray(TPL.build(p)[0]); tp[2] = a.clip; tp[3] = a.refr
        tpls.append(bytes(tp))
    res = dict(model=a.model, clip=a.clip, refr=a.refr, split=a.split, sens=SENS, phrases=PHRASES, pos={}, fa={})
    # positives: eval voices (test) or train voices (dev)
    pos_dir = E if a.split == "test" else R / "data/wake/eval_dev"
    for cond in ("clean", "noise10", "reverb", "far"):
        d = np.load(pos_dir / f"pos_{cond}.npz"); meta = json.load(open(pos_dir / f"pos_{cond}.json"))
        z = f(d["u"]); off = d["off"]
        spans = list(zip(off[:-1], off[1:]))
        counts, events = run_streams(z, spans, tpls, SENS, a.clip, a.refr)
        hit = np.zeros((len(SENS), len(spans), len(tpls)), bool)
        early = np.zeros_like(hit)
        for b, s, ti, t in events:
            m = meta[b]
            if 320 * t + 560 > m["s0"] and 320 * t <= m["s1"] + 9600:
                hit[s, b, ti] = True
            else:
                early[s, b, ti] = True
        out = {}
        for ti, p in enumerate(PHRASES):
            own = np.array([m["phrase"] == p for m in meta])
            other = ~own
            out[p] = dict(n=int(own.sum()), det=hit[:, own, ti].mean(1).round(4).tolist(),
                          other_phrase_fa=(hit[:, other, ti] | early[:, other, ti]).mean(1).round(4).tolist(),
                          outside_window=early[:, own, ti].mean(1).round(4).tolist())
        res["pos"][cond] = out
        print(cond, {p: out[p]["det"][SENS.index(32)] for p in PHRASES}, f"{time.time() - t0:.0f}s", flush=True)
    if not a.skip_fa:
        for name in FA_SPLIT[a.split]:
            d = np.load(E / f"fa_{name}.npz"); z = f(d["u"]); off = d["off"]
            spans = chunks(d["u"], off)
            hours = (off[-1]) * 0.02 / 3600
            counts, _ = run_streams(z, spans, tpls, SENS, a.clip, a.refr)
            per24 = (counts.sum(1) / hours * 24).round(3)        # (n_sens, nt)
            res["fa"][name] = dict(hours=round(hours, 3), count=counts.sum(1).tolist(), per24h=per24.tolist())
            print(name, f"{hours:.1f} h", "fa/24h at sens 32:", per24[SENS.index(32)].tolist(), f"{time.time() - t0:.0f}s", flush=True)
    json.dump(res, open(a.out, "w"))


if __name__ == "__main__":
    main()
