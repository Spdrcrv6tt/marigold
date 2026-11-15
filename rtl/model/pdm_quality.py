"""Signal quality of the PDM front end, measured on the bit-exact reference model.

The RTL is proven sample-identical to this model by rtl/tb/test_pdm.py, so these figures
are the RTL's. The PDM source is the 4th-order sigma-delta model in pdm_model.py (an ideal
modulator; a real MEMS mic is limited by its own acoustic noise, about 61-65 dBA SNR).

Run: python rtl/model/pdm_quality.py     (writes reports/data/phase2-pdm-quality.json)
"""
import json
import os
import sys
import numpy as np
from scipy import signal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pdm_model as pm
import gen_coefs as gc

FS, FS_PDM, NFFT, SKIP = 16000, 3072000, 16384, 1024


F1K = 1021 * FS / NFFT      # "1 kHz": 997.07 Hz, a prime bin so rounding error is not periodic
F2K = 2039 * FS / NFFT      # helper tone for the alias test, 1991.2 Hz


def run_tone(freq, amp, n_out=NFFT + SKIP, dc=0.0, gain=0, dc_en=True, extra=None):
    n_bits = n_out * 192
    t = np.arange(n_bits) / FS_PDM
    u = amp * np.sin(2 * np.pi * freq * t) + dc
    if extra:
        u = u + extra[1] * np.sin(2 * np.pi * extra[0] * t)
    bits = pm.sdm(u)
    y = np.array(pm.pdm_decimate(bits, gain=gain, dc_en=dc_en), dtype=float)
    return y[SKIP:SKIP + NFFT]


def spectrum(y):
    w = signal.windows.blackmanharris(len(y))
    sp = np.abs(np.fft.rfft(y * w)) ** 2
    return sp, w


def band(sp, k, half=4):
    return sp[max(k - half, 0):k + half + 1].sum()


def tone_metrics(y, freq):
    """SNR, THD, SINAD (dB) and amplitude (LSB peak). Coherent bins, Blackman-Harris."""
    sp, w = spectrum(y)
    k0 = int(round(freq * NFFT / FS))
    p_sig = band(sp, k0)
    p_har = 0.0
    mask = np.ones(len(sp), bool)
    mask[:5] = False                      # DC
    mask[k0 - 4:k0 + 5] = False
    for h in range(2, 8):
        k = h * k0
        k = k % NFFT
        if k > NFFT // 2:
            k = NFFT - k
        if 4 < k < NFFT // 2 - 4:
            p_har += band(sp, k)
            mask[k - 4:k + 5] = False
    p_noise = sp[mask].sum()
    amp = np.sqrt(4 * p_sig / (len(y) * (w ** 2).sum()))
    return dict(snr_db=10 * np.log10(p_sig / p_noise), thd_db=10 * np.log10(p_har / p_sig),
                sinad_db=10 * np.log10(p_sig / (p_noise + p_har)), amp_lsb=float(amp))


def tone_level(y, freq):
    sp, w = spectrum(y)
    k0 = int(round(freq * NFFT / FS))
    return float(np.sqrt(4 * band(sp, k0) / (len(y) * (w ** 2).sum())))


def linear_response(f):
    """|H(f)| of CIC * half-band * FIR from the quantised coefficients (ignores rounding)."""
    hb, fir = pm.load_coefs()
    g = gc.cic_mag(f) * gc.fir_mag(np.array(hb) / 32768.0, f, 64000.0) * gc.fir_mag(np.array(fir) / 32768.0, f, 32000.0)
    return g * (gc.R ** gc.N) / 2.0 ** (gc.CIC_W - 1)


def main():
    out = {}
    # --- 1 kHz tone at several levels
    out["tone_1k"] = {}
    for dbfs in (-6, -20, -40, -60):
        amp = 10 ** (dbfs / 20)
        m = tone_metrics(run_tone(F1K, amp), F1K)
        m["level_err_db"] = 20 * np.log10(m["amp_lsb"] / (amp * 32768))
        out["tone_1k"][str(dbfs)] = m
        print("997 Hz %4d dBFS: SNR %.1f dB  THD %.1f dB  SINAD %.1f dB  level error %+.3f dB" % (
            dbfs, m["snr_db"], m["thd_db"], m["sinad_db"], m["level_err_db"]))
    # 16-bit ideal for reference: 6.02*16+1.76 = 98.1 dB at full scale
    # --- idle noise (zero input): RMS in LSB
    bits = pm.sdm(np.zeros((NFFT + SKIP) * 192), dither=0.3)
    y = np.array(pm.pdm_decimate(bits), dtype=float)[SKIP:]
    out["idle_rms_lsb"] = float(np.std(y))
    print("idle noise: %.2f LSB rms (%.1f dBFS)" % (np.std(y), 20 * np.log10(max(np.std(y), 1e-9) / 32768 * np.sqrt(2))))

    # --- passband ripple, analytic from the quantised coefficients
    f = np.linspace(50, 7000, 2000)
    h = 20 * np.log10(linear_response(f))
    out["passband_analytic"] = dict(f_lo=50, f_hi=7000, max_db=float(h.max()), min_db=float(h.min()),
                                    ripple_pp_db=float(h.max() - h.min()))
    print("passband 50 Hz-7 kHz (analytic): %+.3f .. %+.3f dB, ripple %.3f dB p-p" % (h.min(), h.max(), h.max() - h.min()))
    for fe in (7200, 7500, 8000):
        print("  response at %d Hz: %.2f dB" % (fe, 20 * np.log10(linear_response(np.array([float(fe)]))[0])))
        out["resp_%d_db" % fe] = float(20 * np.log10(linear_response(np.array([float(fe)]))[0]))
    # --- passband ripple, measured through modulator + bit-exact model
    meas = {}
    for fr in (250, 500, 1000, 2000, 3000, 4000, 5000, 6000, 6500, 7000):
        fq = round(fr * NFFT / FS) * FS / NFFT
        lv = tone_level(run_tone(fq, 0.25, dc_en=False), fq)
        meas[str(fr)] = 20 * np.log10(lv / (0.25 * 32768))
    out["passband_measured_db"] = meas
    mv = np.array(list(meas.values()))
    out["passband_measured_ripple_pp_db"] = float(mv.max() - mv.min())
    print("passband measured (10 tones 250 Hz-7 kHz, -12 dBFS): %+.3f .. %+.3f dB, ripple %.3f dB p-p" % (
        mv.min(), mv.max(), mv.max() - mv.min()))

    # --- alias rejection, analytic: every input frequency that folds into 0-7 kHz
    worst = (0.0, 0.0)
    fo = np.linspace(0, 7000, 141)
    rej = []
    for m in range(1, 97):                      # images up to the PDM Nyquist (1.536 MHz)
        for sgn in (-1, 1):
            fi = m * 16000 + sgn * fo
            fi = fi[(fi > 0) & (fi <= FS_PDM / 2)]
            if len(fi):
                a = 20 * np.log10(np.maximum(linear_response(fi), 1e-12))
                i = int(np.argmax(a))
                rej.append((float(a[i]), float(fi[i])))
    worst = max(rej)
    out["alias_analytic"] = dict(worst_db=worst[0], at_hz=worst[1], band="folds into 0-7 kHz")
    print("alias rejection (analytic, anything folding into 0-7 kHz): worst %.1f dB at %.0f Hz input" % worst)
    fo2 = np.linspace(0, 6000, 121)
    rej2 = []
    for m in range(1, 97):
        for sgn in (-1, 1):
            fi = m * 16000 + sgn * fo2
            fi = fi[(fi > 0) & (fi <= FS_PDM / 2)]
            if len(fi):
                a = 20 * np.log10(np.maximum(linear_response(fi), 1e-12))
                i = int(np.argmax(a))
                rej2.append((float(a[i]), float(fi[i])))
    w2 = max(rej2)
    out["alias_analytic_0_6k"] = dict(worst_db=w2[0], at_hz=w2[1])
    print("alias rejection (analytic, folding into 0-6 kHz): worst %.1f dB at %.0f Hz input" % w2)
    # --- alias rejection, measured: out-of-band tones through modulator + model
    am = {}
    for fi in (9000.0, 9500.0, 10000.0, 12000.0, 15000.0, 17000.0, 25000.0, 31000.0, 63000.0, 65000.0):
        k = round(fi * NFFT / FS)
        fq = k * FS / NFFT
        fa = abs(fq - round(fq / FS) * FS)
        # a -30 dBFS in-band tone rides along so the alias is not lost in the output rounding
        lv = tone_level(run_tone(fq, 0.5, dc_en=False, extra=(F2K, 0.03)), fa)
        am[str(int(fi))] = dict(alias_hz=fa, rej_db=20 * np.log10(max(lv, 1e-6) / (0.5 * 32768)))
        print("  %6.0f Hz tone at -6 dBFS -> alias at %5.0f Hz: %.1f dB" % (fq, fa, am[str(int(fi))]["rej_db"]))
    out["alias_measured"] = am

    dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "reports", "data")
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(dst, "phase2-pdm-quality.json"), "w", newline="\n") as fjs:
        json.dump(out, fjs, indent=1)


if __name__ == "__main__":
    main()
