"""Shared helpers for Marigold's cocotb tests: random configs and blobs (rtl/model/mg_model.py)."""
import random
import mg_model as M


def rand_cfg(rng, D=None, N=None, n_actions=None, n_slots=None, n_gpio=None, events=True):
    D = D or rng.choice([1, 2, 3, 4, 5, 7, 8, 12, 16, 31, 33, 64, 100, 128, 256])
    na = n_actions or rng.randint(1, 10)
    ns = rng.randint(0, 8) if n_slots is None else n_slots
    ng = rng.randint(0, 32) if n_gpio is None else n_gpio
    wl = rng.randint(0, 64)
    room = 32768 - M.blob_len(D, 0, na, ns, ng, wl)
    nmax = max(1, min(40, room // (D + 4)))
    N = N or rng.randint(1, nmax)
    ids = rng.sample(range(1, 0xF0), na)
    if events and na >= 2:                    # some configs carry event entries
        ne = rng.randint(0, min(3, na - 1))
        for pos, e in zip(rng.sample(range(na), ne), rng.sample([0xF0, 0xF1, 0xF2], ne)):
            ids[pos] = e
    cmd_ids = [i for i in ids if i < 0xF0] or [ids[0]]
    acts = []
    for i in ids:
        gf = rng.randint(0, max(0, ng + 2))
        acts.append((i, rng.randint(0, 255), gf, rng.randint(0, 6)))
    vecs = []
    for _ in range(N):
        a = rng.choice([0] + cmd_ids * 3 + ([rng.randint(1, 0xEF)] if rng.random() < 0.05 else []))
        vecs.append((a, rng.randint(0, 255), rng.randint(-32768, 32767), [rng.randint(-128, 127) for _ in range(D)]))
    slots = []
    for _ in range(ns):
        lo = rng.randint(-1000, 1000)
        lv = tuple(rng.choice([0, 0, rng.randint(1, 255)]) for _ in range(3))
        slots.append((rng.randint(0, 2), lo, lo + rng.randint(-5, 2000), lv))
    gpio = [(rng.randint(0, 255), rng.randint(0, 3), rng.randint(0, 1), rng.randint(0, 9), rng.randint(0, 5))
            for _ in range(ng)]
    return dict(D=D, vectors=vecs, actions=acts, slots=slots, gpio=gpio, wake=bytes(rng.randint(0, 255) for _ in range(wl)),
                host=rng.randint(0, 2), i2c_addr=rng.randint(8, 0x77), reject_thr=0, margin=0,
                wake_sens=rng.randint(0, 65535), flags=rng.randint(0, 15),
                conf_sh=rng.choice([0, 3, 8, 9, 10, 31, 40]))


def tune_thresholds(rng, cfg, q):
    """Pick reject_thr and margin around the scores q gives, so accepts and every kind of reject occur."""
    D = cfg["D"]
    s = sorted(sum(qq * vv for qq, vv in zip(q, v[3])) + v[2] for v in cfg["vectors"])
    cfg["reject_thr"] = rng.choice([s[0] - 1, s[len(s) // 2], s[-1], s[-1] + 1, -(1 << 31), rng.randint(-50000, 50000)])
    cfg["margin"] = rng.choice([0, 0, 1, rng.randint(0, 3000), rng.randint(0, 100000), -5])
    return cfg
