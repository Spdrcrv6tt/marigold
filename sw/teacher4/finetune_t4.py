"""Teacher v4: a bigger, command-tuned SDK text embedder trained directly in the chip's D-space.

Recipe = gate-0b v2/v3 (supervised contrastive loss within one product per batch, tau 0.1, LibriSpeech snippets as
their own classes so ordinary speech stays spread out, non-commands as negatives only), plus:
  * the loss is computed in the chip space: z = W (t - mu), z -= mean(z), z /= |z|, with t the normalised sentence
    embedding and W [D, E] a trained linear head initialised from the PCA of the base model on training text.
    W and mu are saved as <out>/marigold_proj.npz {P, mu}, so golden-format section 5 holds unchanged;
  * opposite-action hard negatives: action pairs whose names differ by an antonym (on/off, warmer/cooler, tilt_up/
    tilt_down, ...) get +log(opp_w) on their negative logits, and phrases are copied with the antonym word swapped
    under the opposite label (minimal pairs);
  * an additive margin on positive logits (--margin, in cosine units);
  * LoRA (--lora r) for large bases, merged before saving; a fixed instruction prompt (--prompt) for Qwen3 models,
    stored as the model's default prompt so every encode() (SDK, targets, eval) uses it.
Only training products (data/gate0b/train_text_v4.json; never fans or microwaves); every line passes the eval leak
filter (sw/textfilter.py: eval/text and eval/text_v2). --holdout leaves products out for model selection.

Usage: python sw/teacher4/finetune_t4.py BASE OUT [--lora 16] [--steps 3000] [--holdout "a,b"]
"""
import argparse, json, math, os, random, re, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(ROOT / "data" / "hf"))
sys.path.insert(0, str(ROOT / "sw"))
from textfilter import EvalFilter  # noqa: E402

ANT = [("on", "off"), ("start", "stop"), ("warmer", "cooler"), ("hotter", "colder"), ("hotter", "cooler"),
       ("warmer", "colder"), ("brighter", "dimmer"), ("brighter", "darker"), ("lighter", "darker"), ("faster", "slower"),
       ("louder", "quieter"), ("open", "close"), ("up", "down"), ("lock", "unlock"), ("next", "previous"),
       ("higher", "lower"), ("extend", "retract"), ("harder", "easier"), ("stronger", "gentler"), ("more", "less"),
       ("high", "low"), ("mute", "unmute"), ("heat", "cool"), ("raise", "lower")]
ANTS = {frozenset(p) for p in ANT}
QWEN_PROMPT = "Instruct: Find the device action the speaker wants, or none\nQuery: "


def opposites(names):
    """{(a, b, word_a, word_b)} for action pairs whose names differ in one '_' token that is an antonym pair."""
    out = []
    for a in names:
        for b in names:
            ta, tb = a.split("_"), b.split("_")
            if a >= b or len(ta) != len(tb):
                continue
            d = [(x, y) for x, y in zip(ta, tb) if x != y]
            if len(d) == 1 and frozenset(d[0]) in ANTS:
                out.append((a, b, d[0][0], d[0][1]))
    return out


def swap(text, w1, w2):
    t = re.sub(rf"\b{w1}\b", "\0", text)
    if t == text:
        return None
    return re.sub(rf"\b{w2}\b", w1, t).replace("\0", w2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base"); ap.add_argument("out")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--lora", type=int, default=0, help="LoRA rank (0 = full fine-tune)")
    ap.add_argument("--per_action", type=int, default=6)
    ap.add_argument("--none", type=int, default=24)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--margin", type=float, default=0.1)
    ap.add_argument("--opp_w", type=float, default=3.0)
    ap.add_argument("--swap_p", type=float, default=0.5)
    ap.add_argument("--libri", type=int, default=32)
    ap.add_argument("--D", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default="data/gate0b/train_text_v4.json")
    ap.add_argument("--holdout", default="")
    ap.add_argument("--prompt", default="auto", help="'auto' = QWEN_PROMPT for Qwen models, none otherwise")
    ap.add_argument("--grad_ckpt", action="store_true", help="(default now; kept for old command lines)")
    ap.add_argument("--no_ckpt", action="store_true", help="disable gradient checkpointing")
    ap.add_argument("--gpu_gb", type=float, default=20.0, help="hard cap on this process's GPU memory (no WDDM spill)")
    ap.add_argument("--min_spm", type=float, default=3.0, help="abort if fewer steps per minute in the first 2 minutes")
    ap.add_argument("--max_len", type=int, default=48)
    a = ap.parse_args()
    random.seed(a.seed); torch.manual_seed(a.seed)
    from sentence_transformers import SentenceTransformer
    prompt = (QWEN_PROMPT if "qwen" in a.base.lower() else "") if a.prompt == "auto" else a.prompt
    flt = EvalFilter()
    hold = {h.strip() for h in a.holdout.split(",") if h.strip()}
    raw = json.load(open(ROOT / a.data, encoding="utf8"))
    data = {p: {"actions": {k: flt.keep(v) for k, v in r["actions"].items()}, "none": flt.keep(r["none"])}
            for p, r in raw.items() if p not in hold}
    print("products", len(data), "held out", sorted(hold), "eval-identical lines dropped", flt.dropped, flush=True)
    pool_none = sorted({t for r in data.values() for t in r["none"]})
    libri = []
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
    prods = [p for p in data if sum(len(v) >= 2 for v in data[p]["actions"].values()) >= 2]
    opp = {p: opposites([k for k, v in data[p]["actions"].items() if len(v) >= 2]) for p in prods}
    print("products with opposite pairs", sum(bool(v) for v in opp.values()), "pairs", sum(len(v) for v in opp.values()), flush=True)

    total = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    torch.cuda.set_per_process_memory_fraction(min(1.0, a.gpu_gb / total))   # OOM instead of spilling into shared memory
    kw = {"dtype": torch.bfloat16} if a.lora else {}
    m = SentenceTransformer(a.base, device="cuda", model_kwargs=kw)
    m.max_seq_length = a.max_len
    tr = m[0]
    if not a.no_ckpt:                       # non-reentrant: gradients reach LoRA weights behind frozen embeddings
        tr.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if a.lora:
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(r=a.lora, lora_alpha=2 * a.lora, lora_dropout=0.05,
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        tr.model = get_peft_model(tr.model, cfg)
        for n_, p_ in tr.model.named_parameters():
            if p_.requires_grad:
                p_.data = p_.data.float()
        tr.model.print_trainable_parameters()

    def embed(texts, grad=True):
        feats = m.tokenize([prompt + t for t in texts])
        feats = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in feats.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16), torch.set_grad_enabled(grad):
            e = m(feats)["sentence_embedding"]
        return torch.nn.functional.normalize(e.float(), dim=-1)

    # head init: PCA of the base model on training text
    m.eval()
    fit = [t for r in data.values() for v in list(r["actions"].values()) + [r["none"]] for t in v]
    random.Random(1).shuffle(fit)
    with torch.no_grad():
        T = torch.cat([embed(fit[i:i + 256], grad=False) for i in range(0, min(len(fit), 12000), 256)]).double()
    mu = T.mean(0)
    _, _, vt = torch.linalg.svd(T - mu, full_matrices=False)
    W = torch.nn.Parameter(vt[:a.D].float().contiguous())
    mu = mu.float()
    m.train()

    def chip(e):
        z = (e - mu) @ W.T
        z = z - z.mean(1, keepdim=True)
        return torch.nn.functional.normalize(z, dim=-1)

    params = [p_ for p_ in m.parameters() if p_.requires_grad]
    opt = torch.optim.AdamW([{"params": params, "lr": a.lr}, {"params": [W], "lr": a.head_lr}], weight_decay=0.01)
    warm = 100
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))
    lw = math.log(a.opp_w)
    import time
    t0, spm_done = time.time(), False
    for step in range(a.steps):
        if not spm_done and time.time() - t0 >= 120:   # throughput sanity line (and guard) after 2 minutes
            spm = step / ((time.time() - t0) / 60); spm_done = True
            print(f"throughput: {step} steps in {time.time() - t0:.0f} s = {spm:.1f} steps/min; GPU max reserved "
                  f"{torch.cuda.max_memory_reserved() / 2 ** 30:.1f} GB (cap {a.gpu_gb} GB)", flush=True)
            if spm < a.min_spm:
                sys.exit(f"ABORT: {spm:.1f} steps/min is below --min_spm {a.min_spm}")
        p = random.choice(prods)
        acts = [k for k, v in data[p]["actions"].items() if len(v) >= 2]
        texts, labels = [], []
        for li, act in enumerate(acts):
            for t in random.sample(data[p]["actions"][act], min(a.per_action, len(data[p]["actions"][act]))):
                texts.append(t); labels.append(li)
        for (x, y, wx, wy) in opp[p]:          # minimal pairs: swap the antonym word, label the opposite action
            ix, iy = acts.index(x), acts.index(y)
            for t, l in list(zip(texts, labels)):
                if l in (ix, iy) and random.random() < a.swap_p:
                    s_ = swap(t, wx, wy) if l == ix else swap(t, wy, wx)
                    if s_ and s_ != t:
                        texts.append(s_); labels.append(iy if l == ix else ix)
        nn = random.sample(data[p]["none"], min(a.none // 2, len(data[p]["none"]))) + random.sample(pool_none, a.none // 2)
        texts += nn; labels += [-1] * len(nn)
        for j, t in enumerate(random.sample(libri, a.libri)):
            texts += [t, t]; labels += [1000 + j, 1000 + j]
        z = chip(embed(texts))
        y = torch.tensor(labels, device="cuda")
        eye = torch.eye(len(y), dtype=torch.bool, device="cuda")
        pos = (y[:, None] == y[None, :]) & ~eye & (y[:, None] >= 0)
        sim = z @ z.T
        sim = (sim - a.margin * pos.float()) / a.tau
        if opp[p]:                              # opposite-action negatives weigh opp_w times
            hard = torch.zeros(len(acts), len(acts), dtype=torch.bool, device="cuda")
            for (x, y_, _, _) in opp[p]:
                hard[acts.index(x), acts.index(y_)] = hard[acts.index(y_), acts.index(x)] = True
            yc = y.clamp(0, len(acts) - 1)
            isact = (y >= 0) & (y < len(acts))
            hm = hard[yc][:, yc] & isact[:, None] & isact[None, :]
            sim = sim + lw * hm.float()
        sim = sim.masked_fill(eye, -1e9)
        logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        anchors = pos.any(1)
        loss = (-(logp * pos).sum(1)[anchors] / pos.sum(1)[anchors]).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params + [W], 1.0)
        opt.step(); sched.step()
        if step % 100 == 0 or step == a.steps - 1:
            print(f"step {step} loss {loss.item():.4f} n {len(texts)}", flush=True)
    m.eval()
    if a.lora:
        tr.model = tr.model.merge_and_unload()
        tr.model.to(torch.bfloat16)
    if prompt:
        m.prompts = {"query": prompt, "document": prompt}
        m.default_prompt_name = "query"
    m.save(a.out)
    np.savez(Path(a.out) / "marigold_proj.npz", mu=mu.detach().cpu().numpy().astype(np.float32),
             P=W.detach().cpu().numpy().astype(np.float32))
    json.dump(vars(a) | {"prompt_used": prompt}, open(Path(a.out) / "marigold_train.json", "w"), indent=1)
    print("saved", a.out, flush=True)


if __name__ == "__main__":
    main()
