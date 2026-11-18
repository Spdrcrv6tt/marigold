"""Marigold live demo: the chip's pipeline on a PC microphone, with a local display at http://127.0.0.1:8899.

    python sw/demo/live.py [--config sdk/configs/fan.json] [--blob data/sdk/fan.mgc] [--model data/intent/qv3.pt]
                           [--mic NAME|INDEX] [--wav FILE [--speed 4]] [--device cpu|cuda] [--exact] [--port 8899]
    python sw/demo/live.py --list                      (input devices)

Per utterance (the PC captures speech with a simple level gate, 0.6 s before it and 0.9 s of quiet after):
the chip's VAD (rtl/model/vad_model.py) -> wake v2 (sw/wake/golden, the blob's template and wake_sens) -> the chip's
segmenter (rtl/model/mg_model.py) -> intent from the QAT torch copy of the chip model (milliseconds; the integer
golden is the chip, bit-exact, but takes seconds) -> the chip's matcher and host frame (mg_model.match / frame).
--exact also runs the integer golden (sw/golden/intent_np.py) in the background and shows whether it agrees.
Every utterance is saved under data/real/live/ (audio + decision): EVALUATION ONLY, never training, never in git.
"""
import argparse, importlib.util, json, os, queue, sys, threading, time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
for p in ("sw/wake", "sw/wake/golden", "rtl/model", "sw/golden", "sw/intent", "sdk"):
    sys.path.insert(0, str(ROOT / p))
os.environ.setdefault("HF_HOME", str(ROOT / "data/hf")); os.environ.setdefault("HF_HUB_OFFLINE", "1")
import mg_model as MG  # noqa: E402
from vad_model import Vad  # noqa: E402
import golden as WG  # noqa: E402
import values as VAL  # noqa: E402
import compile as C  # noqa: E402
import chip_golden as CG  # noqa: E402

SR = 16000
STATE = {"status": "starting", "product": "", "model": "", "device": {}, "last": None, "history": [], "level_db": -90.0}
LOCK = threading.Lock()


def set_state(**kw):
    with LOCK:
        STATE.update(kw)


# ------------------------------------------------------------------ the chip's pipeline on one utterance
class Chip:
    def __init__(self, a):
        self.blob = Path(a.blob).read_bytes()
        err, self.cfg = MG.parse_blob(self.blob)
        if err:
            raise SystemExit(f"blob refused by the chip's loader, error {err}")
        self.b = C.parse(self.blob)
        self.slots = CG.slot_specs(self.blob)
        self.conf = json.load(open(a.config, encoding="utf-8"))
        self.names = {x["id"]: x["name"] for x in self.conf["actions"]}
        self.slot_names = list(self.conf.get("values", {}).keys())
        self.tpl, self.sens = bytes(self.cfg["wake"]), self.cfg["wake_sens"]
        self.P = WG.load_params(str(ROOT / "sw/wake/golden/wake_int_v2.npz"))
        import torch
        self.torch = torch
        spec = importlib.util.spec_from_file_location("intent_model", ROOT / "sw/intent/model.py")
        im = importlib.util.module_from_spec(spec); spec.loader.exec_module(im)
        ck = torch.load(ROOT / a.model if not Path(a.model).is_absolute() else a.model, map_location="cpu")
        m = im.IntentEncoder(ck["cfg"])
        if ck.get("qat"):
            im.enable_qat(m)
        m.load_state_dict(ck["state"]); self.dev = a.device
        self.m = m.to(self.dev).eval()
        self.pool = None
        if a.exact:
            from concurrent.futures import ProcessPoolExecutor
            self.pool = ProcessPoolExecutor(1, initializer=_exact_init, initargs=(str(C.golden_path("v3")),))

    def intent(self, pcm):
        t = self.torch
        x = t.from_numpy((pcm.astype(np.float32) / 32768.0)[None]).to(self.dev)
        with t.no_grad():
            q, lg, fl = self.m(x, t.tensor([len(pcm)], device=self.dev))
        qi = np.clip(np.round(q.float().cpu().numpy()[0] * 32), -127, 127).astype(np.int64)
        toks = VAL.ctc_greedy(lg.float().argmax(-1).cpu().numpy()[0, :int(fl[0])])
        return qi, toks

    def decide(self, qi, toks):
        raw = VAL.raw_values(toks, self.slots); raw = raw + [0] * (8 - len(raw))
        r = MG.match(self.cfg, [int(v) for v in qi], raw)
        best, act, other = C.scores(self.b, qi[None])
        vals = []
        for v in r["values"]:
            k, val = (v[0], v[1]) if isinstance(v, (tuple, list)) else (len(vals), v)
            vals.append((self.slot_names[k] if k < len(self.slot_names) else f"slot{k}", int(val)))
        return dict(ev=int(r["ev"]), action=self.names.get(int(r["action"]), "-") if r["ev"] == MG.EV_COMMAND else None,
                    values=vals, conf=int(r["conf"]), best=int(best[0]), thr=int(self.b["reject_thr"]),
                    closest=self.names.get(int(C.pure_action(self.b, qi[None])[0]), "-"),
                    frame=MG.frame(r["ev"], r["action"], r["values"], r["conf"]).hex(), tokens=toks)

    def run(self, x):
        """x: int16 at 16 kHz. Returns a list of results, one per segment the chip would close."""
        x = np.asarray(x, np.int64)
        t0 = time.time()
        v = Vad().run(x)
        raw, vflag = np.asarray(v["raw"]), np.asarray(v["vad"])
        wakes = WG.run2(x, vflag, self.tpl, self.sens, self.P)["wake_k"]
        if not wakes:
            return [dict(wake=False, ms=int(1000 * (time.time() - t0)))]
        seg, wk = MG.Segmenter(), set(int(w) for w in wakes)
        for w in range(len(x) + 1):
            rp = int(raw[w // 256 - 1]) if (w > 0 and w % 256 == 0 and w // 256 - 1 < len(raw)) else 0
            seg.step(w, w in wk, rp)
        out = []
        for s0, s1, tmo in seg.segs:
            if tmo:
                out.append(dict(wake=True, timeout=True, ms=int(1000 * (time.time() - t0))))
                continue
            qi, toks = self.intent(x[s0:s1])
            d = self.decide(qi, toks)
            d.update(wake=True, timeout=False, segment=(int(s0), int(s1)), ms=int(1000 * (time.time() - t0)))
            if self.pool:
                d["exact"] = "checking"
                d["_future"] = self.pool.submit(_exact_q, x[s0:s1])
            out.append(d)
        if not out:
            out.append(dict(wake=True, timeout=True, ms=int(1000 * (time.time() - t0)), note="segment did not close"))
        return out


_G = {}


def _exact_init(gpath):
    import intent_np as S
    _G["I"] = S.IntIntent(dict(np.load(gpath)))


def _exact_q(pcm):
    q, k = _G["I"].run(np.asarray(pcm, np.int64))
    return np.asarray(q, np.int64), VAL.ctc_greedy(k)


# ------------------------------------------------------------------ a virtual product that reacts
def apply(device, key, action, values):
    v = dict(values)
    if key == "fan":
        if action == "on":
            device.update(power="on")
        elif action == "off":
            device.update(power="off")
        elif action in ("faster", "slower"):
            device.update(power="on", speed=max(1, min(5, device.get("speed", 3) + (1 if action == "faster" else -1))))
        elif action == "set_level" and "speed" in v:
            device.update(power="on", speed=v["speed"])
        elif action == "oscillate":
            device.update(oscillating=not device.get("oscillating", False))
    elif key == "microwave":
        if action == "start":
            device.update(running=True, seconds=v.get("time", device.get("seconds", 30)), power=v.get("power", device.get("power", 10)))
        elif action == "add_30s":
            device.update(running=True, seconds=device.get("seconds", 0) + 30)
        elif action == "stop":
            device.update(running=False)
        elif action == "power_level" and "power" in v:
            device.update(power=v["power"])
    else:
        device.update(last=action)
    return device


# ------------------------------------------------------------------ capture: level gate -> utterances
def utterances(blocks, rate, pre=0.6, post=0.9, max_s=10.0):
    """blocks: iterator of int16 arrays at `rate`. Yields int16 utterances at `rate`."""
    fl = int(0.02 * rate)
    ring, ring_n = [], int(pre * rate)
    cur, quiet, speech_n, hist = None, 0, 0, []
    buf = np.zeros(0, np.int16)
    for blk in blocks:
        buf = np.concatenate([buf, blk])
        while len(buf) >= fl:
            f, buf = buf[:fl], buf[fl:]
            db = 20 * np.log10(np.sqrt(np.mean(f.astype(np.float64) ** 2)) / 32768 + 1e-9)
            hist.append(db); hist[:] = hist[-250:]
            floor = float(np.percentile(hist, 20))
            set_state(level_db=round(db, 1))
            is_sp = db > floor + 9.0 and db > -55.0
            if cur is None:
                ring.append(f)
                while sum(len(r) for r in ring) > ring_n:
                    ring.pop(0)
                if is_sp:
                    cur, quiet, speech_n = list(ring), 0, 0
                    ring = []
                    set_state(status="hearing speech")
            else:
                cur.append(f)
                speech_n += is_sp
                quiet = 0 if is_sp else quiet + 1
                n = sum(len(c) for c in cur)
                if quiet * fl >= post * rate or n >= max_s * rate:
                    if speech_n * fl >= 0.25 * rate:
                        yield np.concatenate(cur)
                    cur = None
                    set_state(status="listening")


def to16k(x, rate):
    if rate == SR:
        return x
    from scipy.signal import resample_poly
    g = gcd(SR, rate)
    y = resample_poly(x.astype(np.float64), SR // g, rate // g)
    return np.clip(np.round(y), -32768, 32767).astype(np.int16)


def mic_blocks(dev, rate):
    import sounddevice as sd
    q = queue.Queue()
    st = sd.InputStream(device=dev, channels=1, samplerate=rate, dtype="int16", blocksize=int(rate * 0.02),
                        callback=lambda d, n, t, s: q.put(d[:, 0].copy()))
    with st:
        while True:
            yield q.get()


def wav_blocks(path, speed):
    x, rate = sf.read(path, dtype="int16")
    x = x if x.ndim == 1 else x[:, 0]
    n = int(rate * 0.02)
    for i in range(0, len(x), n):
        yield x[i:i + n]
        time.sleep(0.02 / speed)
    yield np.zeros(int(rate * 1.5), np.int16)          # trailing quiet closes the last utterance


# ------------------------------------------------------------------ display
PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Marigold live</title><style>
:root{--bg:#f7f7f5;--fg:#1b1b1b;--mut:#6b6b6b;--line:#dcdcd8;--ok:#1d6b3a;--no:#9a3b1e}
@media (prefers-color-scheme:dark){:root{--bg:#141414;--fg:#ececec;--mut:#9a9a9a;--line:#2e2e2e;--ok:#6fcf8f;--no:#f08a6a}}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.4 system-ui,Segoe UI,sans-serif}
main{max-width:860px;margin:0 auto;padding:24px 16px}
h1{font-size:15px;font-weight:600;margin:0 0 4px} .mut{color:var(--mut);font-size:13px}
#status{font-size:14px;margin:18px 0 6px;color:var(--mut)} #big{font-size:40px;font-weight:650;margin:0 0 4px;min-height:52px}
#detail{font-size:14px;color:var(--mut);min-height:20px} #dev{margin:22px 0;padding:14px 0;border-top:1px solid var(--line);border-bottom:1px solid var(--line);font-size:18px}
table{width:100%;border-collapse:collapse;font-size:13px} td,th{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line)} th{color:var(--mut);font-weight:500}
.ok{color:var(--ok)} .no{color:var(--no)}
</style></head><body><main>
<h1 id="product"></h1><div class="mut" id="model"></div>
<div id="status"></div><div id="big"></div><div id="detail"></div>
<div id="dev"></div>
<table><thead><tr><th>Time</th><th>Heard</th><th>Decision</th><th>Score / threshold</th><th>Exact chip</th><th>ms</th></tr></thead><tbody id="hist"></tbody></table>
</main><script>
function fmt(d){if(!d)return '';if(!d.wake)return 'no wake word';if(d.timeout)return 'wake, then nothing';
 return d.action?(d.action+(d.values&&d.values.length?' ('+d.values.map(v=>v[0]+' '+v[1]).join(', ')+')':'')):"didn't catch that"}
async function tick(){try{const s=await (await fetch('/state')).json();
 document.getElementById('product').textContent=s.product;document.getElementById('model').textContent=s.model;
 document.getElementById('status').textContent=s.status+'  ·  input '+s.level_db+' dBFS';
 const d=s.last;const big=document.getElementById('big');big.textContent=fmt(d);big.className=d&&d.action?'ok':(d&&d.wake?'no':'');
 document.getElementById('detail').textContent=d&&d.wake&&!d.timeout?('closest: '+d.closest+'  ·  score '+d.best+' / '+d.thr+'  ·  confidence '+d.conf+(d.exact?'  ·  exact chip: '+d.exact:'')):'';
 document.getElementById('dev').textContent=Object.entries(s.device).map(([k,v])=>k+': '+v).join('    ')||'device: idle';
 document.getElementById('hist').innerHTML=s.history.slice().reverse().map(h=>'<tr><td>'+h.time+'</td><td>'+(h.wake?'wake':'-')+'</td><td class="'+(h.action?'ok':'no')+'">'+fmt(h)+'</td><td>'+(h.best!==undefined?h.best+' / '+h.thr:'')+'</td><td>'+(h.exact||'')+'</td><td>'+h.ms+'</td></tr>').join('');
}catch(e){} setTimeout(tick,300)} tick();
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.startswith("/state"):
            with LOCK:
                body = json.dumps({k: v for k, v in STATE.items()}, default=str).encode()
            ct = "application/json"
        else:
            body, ct = PAGE.encode(), "text/html; charset=utf-8"
        self.send_response(200); self.send_header("Content-Type", ct); self.send_header("Cache-Control", "no-store")
        self.end_headers(); self.wfile.write(body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "sdk/configs/fan.json")); ap.add_argument("--blob", default=None)
    ap.add_argument("--model", default="data/intent/qv3.pt"); ap.add_argument("--device", default="cpu")
    ap.add_argument("--mic", default=None); ap.add_argument("--wav", default=None); ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--exact", action="store_true"); ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--list", action="store_true"); ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args()
    if a.list:
        import sounddevice as sd
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0:
                print(i, sd.query_hostapis(d["hostapi"])["name"], "|", d["name"], int(d["default_samplerate"]))
        return
    key = Path(a.config).stem
    a.blob = a.blob or str(ROOT / f"data/sdk/{key}.mgc")
    chip = Chip(a)
    set_state(product=chip.conf.get("product", key), model=f"model {a.model} · blob {a.blob} · {a.device}"
              + (" · exact check on" if a.exact else ""), status="listening")
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"display: http://127.0.0.1:{a.port}", flush=True)
    if a.wav:
        rate = sf.info(a.wav).samplerate; blocks = wav_blocks(a.wav, a.speed)
    else:
        import sounddevice as sd
        dev = int(a.mic) if (a.mic or "").isdigit() else a.mic
        rate = int(sd.query_devices(dev, "input")["default_samplerate"]); blocks = mic_blocks(dev, rate)
    save = ROOT / "data/real/live"; save.mkdir(parents=True, exist_ok=True)
    device, pending, agree = {}, [], (0, 0)
    for utt in utterances(blocks, rate):
        set_state(status="thinking")
        x = to16k(utt, rate)
        for d in chip.run(x):
            fut = d.pop("_future", None)
            d["time"] = datetime.now().strftime("%H:%M:%S")
            if d.get("action"):
                device = apply(device, key, d["action"], d["values"])
            if fut is not None:
                pending.append((d, fut))
            with LOCK:
                STATE["history"].append(d); STATE["history"] = STATE["history"][-40:]
                STATE["last"], STATE["device"] = d, dict(device)
            print(f"{d['time']} {('no wake' if not d['wake'] else 'wake, nothing' if d.get('timeout') else (d.get('action') or 'rejected'))}"
                  f" {d.get('values', '')} best {d.get('best', '')} thr {d.get('thr', '')} closest {d.get('closest', '')} {d['ms']} ms", flush=True)
            if not a.no_save:
                stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
                sf.write(str(save / f"{stamp}.wav"), x, SR, subtype="PCM_16")
                with open(save / "log.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(dict(d, wav=f"{stamp}.wav", model=a.model, blob=a.blob), default=str) + "\n")
        agree = resolve(chip, pending, agree, block=False)    # exact chip check: integer golden in the background
        set_state(status="listening")
    agree = resolve(chip, pending, agree, block=True)
    if a.exact and sum(agree):
        print(f"exact chip check: {agree[0]} of {sum(agree)} decisions identical", flush=True)
    if a.wav:
        print("end of file", flush=True)


def resolve(chip, pending, agree=(0, 0), block=False):
    """Finish exact checks that are done (or all, if block): (same, different) counts."""
    same, diff = agree
    for d, fut in list(pending):
        if block or fut.done():
            qe, te = fut.result()
            e = chip.decide(qe, te)
            ok = (e["action"], e["values"]) == (d.get("action"), d.get("values"))
            d["exact"] = "same" if ok else f"differs: {e['action'] or 'rejected'}"
            same, diff = same + ok, diff + (not ok)
            print(f"  exact check {d['time']}: {d['exact']} (fast {d.get('action') or 'rejected'}, score {d.get('best')} vs exact {e['best']})", flush=True)
            pending.remove((d, fut))
    return same, diff


if __name__ == "__main__":
    main()
