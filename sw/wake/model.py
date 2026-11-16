"""Wake phone model: causal depthwise-separable 1D CNN on the integer front end's uint8 codes.

Float model with BatchNorm for training; `fold()` gives the BN-folded convs used by QAT and export.
Input per 20 ms frame: 64 codes (two 10 ms frames x 32 mel bands), x = (u - 128) / 32.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

C, K_DW, DIL, N_IN, N_OUT = 96, 5, (1, 1, 2, 2, 2), 64, 40


class Net(nn.Module):
    def __init__(self, c=C, dil=DIL, n_out=N_OUT):
        super().__init__()
        self.dil = dil
        self.l0 = nn.Conv1d(N_IN, c, 1, bias=False)
        self.b0 = nn.BatchNorm1d(c)
        self.dw = nn.ModuleList([nn.Conv1d(c, c, K_DW, groups=c, dilation=d, bias=False) for d in dil])
        self.bd = nn.ModuleList([nn.BatchNorm1d(c) for _ in dil])
        self.pw = nn.ModuleList([nn.Conv1d(c, c, 1, bias=False) for _ in dil])
        self.bp = nn.ModuleList([nn.BatchNorm1d(c) for _ in dil])
        self.out = nn.Conv1d(c, n_out, 1)

    def forward(self, u):
        """u: (B, T, 64) codes (any numeric dtype). Returns logits (B, T, 40)."""
        x = (u.float() - 128.0) / 32.0
        x = x.transpose(1, 2)
        y = F.relu(self.b0(self.l0(x)))
        for dw, bd, pw, bp, d in zip(self.dw, self.bd, self.pw, self.bp, self.dil):
            y = F.relu(bd(dw(F.pad(y, ((K_DW - 1) * d, 0)))))
            y = F.relu(bp(pw(y)))
        return self.out(y).transpose(1, 2)


def n_weights(m):
    return sum(p.numel() for n, p in m.named_parameters() if p.dim() > 1)


def fold(m):
    """List of layers (kind, W, b, dil) with BN folded: kind in {pw, dw, out}; W (Co, Ci or 1, k)."""
    def f(conv, bn):
        s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        W = conv.weight * s[:, None, None]
        b = bn.bias - bn.running_mean * s
        return W.detach().clone(), b.detach().clone()
    L = [("pw", *f(m.l0, m.b0), 1)]
    for dw, bd, pw, bp, d in zip(m.dw, m.bd, m.pw, m.bp, m.dil):
        L.append(("dw", *f(dw, bd), d))
        L.append(("pw", *f(pw, bp), 1))
    L.append(("out", m.out.weight.detach().clone(), m.out.bias.detach().clone(), 1))
    return L


if __name__ == "__main__":
    m = Net()
    print("weights", n_weights(m))
    print(m(torch.randint(0, 255, (2, 50, 64))).shape)
