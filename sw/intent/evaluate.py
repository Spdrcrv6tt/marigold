"""Synthetic-audio evaluation of the Marigold intent model (targets 1, 2, 3, 4, 10, 11).

  python sw/intent/evaluate.py --model data/intent/f1.pt [--qat] [--golden data/golden/mg-v1.npz] --tag f1

Audio: data/tts/eval (held-out TTS voices; paraphrases dev/test, non-commands dev/test, typed config phrases) and
its 10 dB fan and microwave copies (data/tts/eval_noisy). SYNTHETIC ONLY: no real recordings exist yet.
Configs: sdk/configs/{fan,microwave}.json compiled by sdk/compile.py with teacher-v3 and its projection
(golden-format section 5), prototypes from the typed examples plus the SDK expansion, 16 reject prototypes.
One model (same weights) serves both configs (target 11).
Threshold rule (iteration-1.md): per product, the lowest reject threshold that rejects 95% of the CLEAN DEV headline
non-commands (chat, near, tv); written into the blob and used unchanged for test, typed and noisy audio.
Accuracy counts a rejection as an error. Values (target 3): exact match of every labelled slot, on test
paraphrases with values, (a) given the right action and (b) end to end (action right and values right).
"""
import argparse, json, os, sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(ROOT / "data" / "hf"))
sys.path.insert(0, str(ROOT / "sw" / "intent")); sys.path.insert(0, str(ROOT / "sdk")); sys.path.insert(0, str(ROOT / "sw" / "golden"))
import values as VAL  # noqa: E402
import compile as C  # noqa: E402

HEAD = ("chat", "near", "tv")
CONDS = {"clean": "data/tts/eval/manifest.jsonl", "fan10db": "data/tts/eval_noisy/fan10db/manifest.jsonl",
         "microwave10db": "data/tts/eval_noisy/microwave10db/manifest.jsonl"}


def load_rows(path):
    return [json.loads(l) for l in open(ROOT / path, encoding="utf8")]


def run_torch(m, paths, dev="cuda", bs=32):
    qs, toks = [], []
    for i in range(0, len(paths), bs):
        xs = [sf.read(str(ROOT / p), dtype="float32")[0] for p in paths[i:i + bs]]
        n = max(len(x) for x in xs)
        pcm = np.zeros((len(xs), n), np.float32)
        for j, x in enumerate(xs):
            pcm[j, :len(x)] = np.round(x * 32767) / 32768
        with torch.no_grad():
            q, lg, fl = m(torch.from_numpy(pcm).to(dev), torch.tensor([len(x) for x in xs], device=dev))
        qs.append(np.clip(np.round(q.float().cpu().numpy() * 32), -127, 127).astype(np.int32))
        k = lg.float().argmax(-1).cpu().numpy()
        toks += [VAL.ctc_greedy(k[j, :int(fl[j])]) for j in range(len(xs))]
    return np.concatenate(qs), toks


def _gold_worker(args):
    gpath, paths = args
    import intent_np as S
    G = dict(np.load(gpath))
    I = S.IntIntent(G)
    out = []
    for p in paths:
        x = sf.read(str(ROOT / p), dtype="int16")[0].astype(np.int64)
        q, k = I.run(x)
        out.append((q.astype(np.int32), VAL.ctc_greedy(k)))
    return out


def run_golden(gpath, paths, procs=12):
    from multiprocessing import Pool
    chunks = [paths[i::procs] for i in range(procs)]
    with Pool(procs) as pool:
        res = pool.map(_gold_worker, [(gpath, c) for c in chunks])
    q = [None] * len(paths); t = [None] * len(paths)
    for i, r in enumerate(res):
        for j, (qq, tt) in enumerate(r):
            q[i + j * procs] = qq; t[i + j * procs] = tt
    return np.stack(q), t


def decide(b, Q, thr):
    """Matching rule (config-blob.md, arch 8.2) with threshold thr: returns (action or 0, best score)."""
    S = Q @ b["vectors"].astype(np.int32).T + b["bias"][None]
    bi = S.argmax(1); best = S[np.arange(len(S)), bi]; act = b["vec_action"][bi]
    out = np.where((act == 0) | (best < thr), 0, act)
    if b["margin"] > 0:
        for i in range(len(S)):
            other = S[i][b["vec_action"] != act[i]]
            if out[i] and other.size and best[i] - other.max() < b["margin"]:
                out[i] = 0
    return out, best


def slot_values(b, cfg, action_id, toks):
    a = next(x for x in b["actions"] if x["id"] == action_id)
    idx = [k for k in range(len(b["slots"])) if a["slot_mask"] >> k & 1]
    names = list(cfg.get("values", {}))
    sl = [dict(type=b["slots"][k]["type"], min=b["slots"][k]["min"], max=b["slots"][k]["max"],
               words={w: b["slots"][k][w] for w in ("low", "medium", "high") if b["slots"][k][w]}) for k in idx]
    return {names[k]: v for k, v in zip(idx, VAL.parse(toks, sl))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--qat", action="store_true")
    ap.add_argument("--golden", default="")
    ap.add_argument("--teacher", default="data/models/teacher-v3")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--conds", default="clean,fan10db,microwave10db")
    a = ap.parse_args()
    rows = {c: load_rows(CONDS[c]) for c in a.conds.split(",")}
    rows = {c: [r for r in rs if r["product"] in (None, "fan", "microwave") or r["set"] == "noncommands"] for c, rs in rows.items()}
    # model outputs per condition
    out = {}
    if a.golden:
        for c, rs in rows.items():
            out[c] = run_golden(str(ROOT / a.golden), [r["path"] for r in rs])
    else:
        from model import IntentEncoder, enable_qat
        ck = torch.load(ROOT / a.model, map_location="cuda")
        m = IntentEncoder(ck["cfg"])
        if a.qat or ck.get("qat"):
            enable_qat(m)
        m.load_state_dict(ck["state"]); m.cuda().eval()
        for c, rs in rows.items():
            out[c] = run_torch(m, [r["path"] for r in rs])
    emb = C.STEmbedder(ROOT / a.teacher)
    exp = json.load(open(ROOT / "data/expand/expanded_qwen25.json", encoding="utf8"))
    res = {"model": a.model, "golden": a.golden or None, "synthetic_only": True, "products": {}}
    for prod in ("fan", "microwave"):
        cfg = json.load(open(ROOT / f"sdk/configs/{prod}.json", encoding="utf8"))
        blob, rep = C.compile_config(cfg, prod, emb, exp, proj="teacher")
        b = C.parse(blob)
        name2id = {x["name"]: x["id"] for x in cfg["actions"]}
        # threshold on clean dev headline non-commands
        rs = rows["clean"]; Q, _ = out["clean"]
        dv = [i for i, r in enumerate(rs) if r["set"] == "noncommands" and r["split"] == "dev" and r["category"] in HEAD]
        _, best = decide(b, Q[dv], -10 ** 9)
        acts, _ = decide(b, Q[dv], -10 ** 9)
        open_best = np.sort(best[acts != 0])
        k = int(np.ceil(0.95 * len(dv)))                 # reject >= 95%: at most len - k may pass
        allowed = len(dv) - k
        thr = int(open_best[::-1][allowed]) + 1 if len(open_best) > allowed else -10 ** 9
        b["reject_thr"] = thr
        blob = C.pack({k_: b[k_] for k_ in ("D", "reject_thr", "margin", "wake_sens", "flags", "host", "i2c_addr", "conf_sh")},
                      b["vec_action"], b["vec_flags"], b["bias"], b["vectors"], b["actions"], b["slots"], b["gpio"], b["wake"])
        outdir = ROOT / f"data/sdk/eval-{a.tag}"; outdir.mkdir(parents=True, exist_ok=True)
        (outdir / f"{prod}.mgc").write_bytes(blob)
        pr = {"N": int(b["N"]), "D": int(b["D"]), "reject_thr": thr, "conds": {}}
        for c, rs in rows.items():
            Q, T = out[c]
            pred, best = decide(b, Q, thr)
            sel = lambda f: np.array([i for i, r in enumerate(rs) if f(r)], dtype=int)
            para = sel(lambda r: r["set"] == "paraphrases" and r["product"] == prod and r["split"] == "test")
            typed = sel(lambda r: r["set"] == f"config:{prod}")
            nonc = sel(lambda r: r["set"] == "noncommands" and r["split"] == "test" and r["category"] in HEAD)
            other = sel(lambda r: r["set"] == "noncommands" and r["split"] == "test" and r["category"] == "other_device")
            y = lambda ix: np.array([name2id[rs[i]["action"]] for i in ix])
            d = {}
            d["t2_unseen_acc"] = float((pred[para] == y(para)).mean())
            d["t2_unseen_acc_norej"] = float((decide(b, Q[para], -10 ** 9)[0] == y(para)).mean())
            d["t1_typed_acc"] = float((pred[typed] == y(typed)).mean()) if len(typed) else None
            d["t4_reject_headline"] = float((pred[nonc] == 0).mean())
            d["reject_other_device"] = float((pred[other] == 0).mean())
            d["n"] = {"para": len(para), "typed": len(typed), "nonc": len(nonc)}
            vr = [i for i in para if rs[i]["values"]]
            ok_given, ok_e2e = [], []
            for i in vr:
                got = slot_values(b, cfg, name2id[rs[i]["action"]], T[i])
                good = all(got.get(kk) == vv for kk, vv in rs[i]["values"].items())
                ok_given.append(good); ok_e2e.append(good and pred[i] == name2id[rs[i]["action"]])
            d["t3_values_given_action"] = float(np.mean(ok_given)) if vr else None
            d["t3_values_end_to_end"] = float(np.mean(ok_e2e)) if vr else None
            d["n"]["values"] = len(vr)
            per = {}
            for nm, aid in name2id.items():
                ii = [i for i in para if rs[i]["action"] == nm]
                per[nm] = float((pred[ii] == aid).mean()) if ii else None
            d["per_action_test"] = per
            pr["conds"][c] = d
            print(f"{prod:<9} {c:<13} unseen {d['t2_unseen_acc']:.3f} (no-rej {d['t2_unseen_acc_norej']:.3f}) typed "
                  f"{d['t1_typed_acc'] if d['t1_typed_acc'] is None else round(d['t1_typed_acc'], 3)} reject {d['t4_reject_headline']:.3f} "
                  f"other-dev {d['reject_other_device']:.2f} values {d['t3_values_given_action']} / e2e {d['t3_values_end_to_end']}", flush=True)
        res["products"][prod] = pr
    Path(ROOT / "reports/p0").mkdir(parents=True, exist_ok=True)
    json.dump(res, open(ROOT / f"reports/p0/eval-{a.tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
