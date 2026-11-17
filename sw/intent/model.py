"""Marigold intent encoder (float/QAT reference). The graph is Poinsetta's streaming Moonshine encoder, shrunk,
with the same op set (docs/golden-format.md):

  pcm -> 80-sample frames -> per-frame norm -> asinh(k x) -> fe.lin + SiLU -> causal conv k5 s2 + SiLU
      -> causal conv k5 s2 (50 frames/s) -> NL x [pre-norm band attention, pre-norm GELU MLP] -> enc.ln
      -> e[t] (per frame)
  intent: q = norm(proj(mean_t e[t])) (no gain, no bias) -> int8 in the golden model
  values: CTC logits = ctc(e[t]) per frame

Streaming: the convs are causal; layer 0 looks FUT0 frames ahead (4 frames = one 80 ms chunk), the others none,
so a frame's output depends on at most 80 ms of later audio.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

FRAME = 80
# v1 (Nov 15, 7 PM): Moonshine Streaming Tiny's own frontend and its first two encoder layers (MIT, pretrained on
# far more speech than we have), so the 4.6M-weight encoder starts from real acoustic features. The from-scratch
# W = 240 design (CFG_SCRATCH) barely learned in 5k steps (cosine 0.12) and was dropped.
CFG = dict(W=320, NH=8, HD=40, NL=2, FF=1280, C1=640, D=128, PAST=16, FUT=(4, 0), V=45)
CFG_SCRATCH = dict(W=240, NH=6, HD=40, NL=6, FF=480, C1=240, D=128, PAST=16, FUT=(4, 0, 0, 0, 0, 0), V=45)
MOONSHINE = "D:/poinsetta/data/hf/hub/models--UsefulSensors--moonshine-streaming-tiny"


def causal_patches(x, k=5, s=2):
    """x [B, N, C] -> [B, ceil(N/s), C*k], channel-major columns (c*k + tap), k-1 zero frames in front."""
    B, N, C = x.shape
    xp = F.pad(x, (0, 0, k - 1, 0))
    n_out = (N + s - 1) // s
    need = (n_out - 1) * s + k
    if xp.shape[1] < need:
        xp = F.pad(xp, (0, 0, 0, need - xp.shape[1]))
    p = xp.unfold(1, k, s)[:, :n_out]            # [B, n_out, C, k]
    return p.reshape(B, n_out, C * k)


class Norm(nn.Module):
    """LayerNorm over the last axis with a gain and no offset (golden 'norm')."""
    def __init__(self, n, eps=1e-5, gain=True):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(n)) if gain else None

    def forward(self, x):
        m = x.mean(-1, keepdim=True); c = x - m
        y = c * torch.rsqrt((c * c).mean(-1, keepdim=True) + self.eps)
        return y * self.g if self.g is not None else y


class Layer(nn.Module):
    def __init__(self, W, NH, HD, FF, past, fut):
        super().__init__()
        self.NH, self.HD, self.past, self.fut = NH, HD, past, fut
        self.ln1, self.ln2 = Norm(W), Norm(W)
        self.q, self.k, self.v, self.o = (nn.Linear(W, W, bias=False) for _ in range(4))
        self.fc1, self.fc2 = nn.Linear(W, FF), nn.Linear(FF, W)

    def forward(self, x, mask_len):
        B, T, W = x.shape
        y = self.ln1(x)
        q = self.q(y).view(B, T, self.NH, self.HD).transpose(1, 2)
        k = self.k(y).view(B, T, self.NH, self.HD).transpose(1, 2)
        v = self.v(y).view(B, T, self.NH, self.HD).transpose(1, 2)
        i = torch.arange(T, device=x.device)
        band = (i[None, :] >= i[:, None] - self.past) & (i[None, :] <= i[:, None] + self.fut)   # [T, T] key in window
        valid = band[None] & (i[None, None, :] < mask_len[:, None, None])                       # [B, T, T]
        valid = valid | torch.eye(T, dtype=torch.bool, device=x.device)[None]   # padding rows keep one key (no NaN)
        s = (q @ k.transpose(-1, -2)).float() * (self.HD ** -0.5)
        s = s.masked_fill(~valid[:, None], float("-inf"))
        a = torch.softmax(s, -1).to(v.dtype) @ v
        x = x + self.o(a.transpose(1, 2).reshape(B, T, W))
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


class IntentEncoder(nn.Module):
    def __init__(self, cfg=CFG):
        super().__init__()
        c = self.cfg = dict(cfg)
        W = c["W"]
        self.log_k = nn.Parameter(torch.tensor(math.log(0.6448)))
        self.fe_lin = nn.Linear(FRAME, W, bias=False)
        C1 = c.get("C1", W)
        self.fe_c1 = nn.Linear(W * 5, C1)
        self.fe_c2 = nn.Linear(C1 * 5, W)
        self.layers = nn.ModuleList(Layer(W, c["NH"], c["HD"], c["FF"], c["PAST"], c["FUT"][i]) for i in range(c["NL"]))
        self.ln = Norm(W)
        self.proj = nn.Linear(W, c["D"], bias=False)
        self.out_norm = Norm(c["D"], gain=False)
        self.ctc = nn.Linear(W, c["V"])

    def frames(self, n_samples):
        """Encoder frames for a segment of n samples (ceil(ceil(n/80)/4))."""
        n = (n_samples + FRAME - 1) // FRAME
        return (n + 3) // 4

    def encode(self, pcm, lens):
        """pcm [B, N] float in [-1, 1), lens [B] samples -> e [B, T, W], frame lengths [B]."""
        B, N = pcm.shape
        pad = (-N) % FRAME
        x = F.pad(pcm, (0, pad)).view(B, -1, FRAME)
        m = x.mean(-1, keepdim=True); c = x - m
        y = c * torch.rsqrt((c * c).mean(-1, keepdim=True) + 1e-6)
        y = torch.asinh(torch.exp(self.log_k) * y)
        h = F.silu(self.fe_lin(y))
        h = F.silu(self.fe_c1(causal_patches(h)))
        x = self.fe_c2(causal_patches(h))
        flen = torch.div(torch.div(lens + FRAME - 1, FRAME, rounding_mode="floor") + 3, 4, rounding_mode="floor")
        for L in self.layers:
            x = L(x, flen)
        return self.ln(x), flen

    def pool(self, e, flen):
        T = e.shape[1]
        m = (torch.arange(T, device=e.device)[None] < flen[:, None]).float()
        s = (e * m[..., None]).sum(1) / flen[:, None].float()
        return self.out_norm(self.proj(s))          # unit RMS, zero mean over D

    def forward(self, pcm, lens):
        e, flen = self.encode(pcm, lens)
        return self.pool(e, flen), self.ctc(e), flen


def load_moonshine(m):
    """Copy Moonshine Streaming Tiny's frontend, first NL encoder layers and final norm into m (read-only file)."""
    import glob
    from safetensors import safe_open
    f = glob.glob(MOONSHINE + "/snapshots/*/model.safetensors")[0]
    sd = {}
    with safe_open(f, "pt") as t:
        g = lambda k: t.get_tensor("model.encoder." + k).float()
        sd["log_k"] = g("embedder.comp.log_k")
        sd["fe_lin.weight"] = g("embedder.linear.weight")
        sd["fe_c1.weight"] = g("embedder.conv1.weight").reshape(m.fe_c1.weight.shape); sd["fe_c1.bias"] = g("embedder.conv1.bias")
        sd["fe_c2.weight"] = g("embedder.conv2.weight").reshape(m.fe_c2.weight.shape); sd["fe_c2.bias"] = g("embedder.conv2.bias")
        for i in range(len(m.layers)):
            L = f"layers.{i}."
            for c in "qkvo":
                sd[L + c + ".weight"] = g(L + f"self_attn.{c}_proj.weight")
            for c in ("fc1", "fc2"):
                sd[L + c + ".weight"] = g(L + f"mlp.{c}.weight"); sd[L + c + ".bias"] = g(L + f"mlp.{c}.bias")
            sd[L + "ln1.g"] = g(L + "input_layernorm.gamma") + 1.0
            sd[L + "ln2.g"] = g(L + "post_attention_layernorm.gamma") + 1.0
        sd["ln.g"] = g("final_norm.gamma") + 1.0
    missing = m.load_state_dict(sd, strict=False)
    return sorted(sd), missing


def n_weights(m):
    return sum(p.numel() for n, p in m.named_parameters() if p.dim() == 2)


if __name__ == "__main__":
    m = IntentEncoder()
    print("weights (matrices)", n_weights(m), "all params", sum(p.numel() for p in m.parameters()))
    x = torch.randn(2, 16000 * 3) * 0.1
    q, lg, fl = m(x, torch.tensor([48000, 30000]))
    print(q.shape, lg.shape, fl)


# ---------------------------------------------------------------------------------------------- QAT (weights only)
BITS8 = ("fe_lin", "ctc")     # 8-bit sites (small); every other matrix 4-bit so the ROM stays under 20 Mbit


def mse_scale(w, bits, grid=48):
    """Per-row scale minimising the squared rounding error (Poinsetta sw/quant.py, mode 'mse')."""
    qmax = 2 ** (bits - 1) - 1
    mx = w.abs().amax(1).clamp_min(1e-12)
    best, scale = torch.full_like(mx, float("inf")), mx / qmax
    for r in torch.linspace(0.25, 1.0, grid, device=w.device):
        s = mx * r / qmax
        err = (((w / s[:, None]).round().clamp(-qmax, qmax) * s[:, None] - w) ** 2).sum(1)
        better = err < best
        best = torch.where(better, err, best); scale = torch.where(better, s, scale)
    return scale


class QLinear(nn.Linear):
    """nn.Linear whose weight passes a per-row fake quantiser (straight-through), log-scales learned."""
    def forward(self, x):
        qmax = 2 ** (self.bits - 1) - 1
        s = self.log_s.exp()[:, None]
        z = (self.weight / s).clamp(-qmax, qmax)
        return F.linear(x, (z + (z.round() - z).detach()) * s, self.bias)


def enable_qat(m):
    for name, mod in m.named_modules():
        if type(mod) is nn.Linear:
            mod.bits = 8 if name in BITS8 else 4
            mod.log_s = nn.Parameter(torch.log(mse_scale(mod.weight.data.float(), mod.bits)))
            mod.__class__ = QLinear
    return m


def quantized(m):
    """{site: (q int8 [out, in], scale [out], bias or None, bits)} from a QAT model."""
    out = {}
    for name, mod in m.named_modules():
        if isinstance(mod, QLinear):
            qmax = 2 ** (mod.bits - 1) - 1
            s = mod.log_s.detach().exp().double()
            q = (mod.weight.detach().double() / s[:, None]).round().clamp(-qmax, qmax).to(torch.int8)
            out[name] = (q.cpu().numpy(), s.cpu().numpy(), None if mod.bias is None else mod.bias.detach().double().cpu().numpy(), mod.bits)
    return out
