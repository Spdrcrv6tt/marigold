"""Markdown tables from kws_eval.py results: detection per condition and false wakes per 24 h per source over wake_sens.

  summarize.py RESULT.json [--phrases "hey breeze,hey kitchen"] [--sens 8,12,16,20,24] [--pick DEV.json]
--pick: choose the operating point on a dev result: the largest wake_sens whose pooled dev false-wake rate (all dev
sources, hours-weighted, mean over the phrases) is at most 1 per 24 h.
"""
import argparse, json

CONDS = ("clean", "noise10", "reverb", "far")


def pooled_fa(r, k, ti):
    h = sum(v["hours"] for v in r["fa"].values())
    c = sum(v["count"][k][ti] for v in r["fa"].values())
    return c / h * 24 if h else float("nan")


def pick(dev, phrases, target=1.0):
    best = None
    for k, s in enumerate(dev["sens"]):
        fa = sum(pooled_fa(dev, k, dev["phrases"].index(p)) for p in phrases) / len(phrases)
        if fa <= target:
            best = s
    return best


def table(r, phrases, sens):
    out = []
    srcs = list(r["fa"])
    head = "| phrase | wake_sens | " + " | ".join(CONDS) + " | " + " | ".join(f"{s} ({r['fa'][s]['hours']:.1f} h)" for s in srcs) + " | pooled FA/24 h |"
    out.append(head)
    out.append("|" + "---|" * (3 + len(CONDS) + len(srcs)))
    for p in phrases:
        ti = r["phrases"].index(p)
        for s in sens:
            k = r["sens"].index(s)
            det = [f"{100 * r['pos'][c][p]['det'][k]:.1f}%" for c in CONDS]
            fa = [f"{r['fa'][n]['per24h'][k][ti]:.1f}" for n in srcs]
            out.append(f"| {p} | {s} | " + " | ".join(det) + " | " + " | ".join(fa) + f" | {pooled_fa(r, k, ti):.1f} |")
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("result")
    ap.add_argument("--phrases", default="hey breeze,hey kitchen")
    ap.add_argument("--sens", default="4,8,12,16,20,24")
    ap.add_argument("--pick", default=None)
    a = ap.parse_args()
    r = json.load(open(a.result))
    ph = a.phrases.split(",")
    sens = [int(x) for x in a.sens.split(",")]
    if a.pick:
        d = json.load(open(a.pick))
        op = pick(d, ph)
        print(f"operating point from dev: wake_sens = {op}")
        if op is not None and op not in sens:
            sens = sorted(sens + [op])
    print(table(r, ph, sens))
    n = {c: r["pos"][c][ph[0]]["n"] for c in CONDS}
    print(f"\npositives per condition: {n}; other-phrase false accepts (row phrase's template on the other phrases' clips) "
          f"at wake_sens {sens[len(sens) // 2]}: " + ", ".join(f"{p} {100 * r['pos']['clean'][p]['other_phrase_fa'][r['sens'].index(sens[len(sens) // 2])]:.1f}%" for p in ph))
