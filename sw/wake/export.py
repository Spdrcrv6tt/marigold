"""Export a QAT checkpoint to the integer golden format and the RTL ROM images.

  export.py QAT_CKPT OUT_DIR
Writes OUT_DIR/wake_int.npz (golden parameters), rtl/wake/mem/wake_rom.mem (weights, 16-bit words in
the engine's read order, docs/golden-wake.md section ROM) and rtl/wake/mem/wake_fe_*.mem (front-end tables).
"""
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fe
from qat import QNet, export
from lexicon import R

MEM = R / "rtl/wake/mem"


def rom_words(L):
    w = []
    for l in L:
        W, b, M, S = l["W"].astype(np.int64), l["b"], l["M"], l["S"]
        Co = W.shape[0]
        assert Co % 4 == 0
        w.append(S)                                         # layer header: shift
        for g in range(Co // 4):
            for j in range(4):
                bb = int(b[4 * g + j]) & 0xFFFFFFFF
                w += [bb & 0xFFFF, bb >> 16]
            n = W.shape[2] if l["kind"] == "dw" else W.shape[1]
            for i in range(n):
                word = 0
                for j in range(4):
                    v = W[4 * g + j, 0, i] if l["kind"] == "dw" else W[4 * g + j, i, 0]
                    word |= (int(v) & 0xF) << (4 * j)
                w.append(word)
            w += [int(M[4 * g + j]) for j in range(4)]
    return w


def write_mem(path, vals, width_hex):
    with open(path, "w") as f:
        for v in vals:
            f.write(f"{int(v):0{width_hex}x}\n")


def fe_tables():
    MEM.mkdir(parents=True, exist_ok=True)
    write_mem(MEM / "wake_fe_win.mem", fe.WIN_Q15[:200], 4)                    # w[i], i < 200; w[399-i] = w[i]
    write_mem(MEM / "wake_fe_sin.mem", fe.SIN_Q, 4)                            # 129 entries
    mel = [(int(fe.MEL_I[k]) << 9) | int(fe.MEL_WU[k]) for k in range(fe.K_MIN, fe.K_MAX + 1)]
    write_mem(MEM / "wake_fe_mel.mem", mel, 4)                                 # bins K_MIN..K_MAX: {band i (6b), wu (9b)}


def main(ckpt, out_dir):
    q = QNet(ckpt)
    L = export(q)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    d = dict(n_layers=len(L))
    for i, l in enumerate(L):
        d.update({f"kind{i}": l["kind"], f"dil{i}": l["dil"], f"W{i}": l["W"], f"b{i}": l["b"], f"M{i}": l["M"], f"S{i}": l["S"]})
    np.savez(out / "wake_int.npz", **d)
    words = rom_words(L)
    MEM.mkdir(parents=True, exist_ok=True)
    write_mem(MEM / "wake_rom.mem", words, 4)
    fe_tables()
    nb = sum(int(np.abs(l["b"]).max()).bit_length() for l in L)
    print(f"layers {len(L)}, ROM words {len(words)} ({len(words) * 16 / 1000:.1f} kbit), shifts {[l['S'] for l in L]}, "
          f"max |b| bits {max(int(np.abs(l['b']).max()).bit_length() for l in L)}, max M {max(int(l['M'].max()) for l in L)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
