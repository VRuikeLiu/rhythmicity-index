"""Score ALL 600 epochs under V1/V2/V3 + baselines + specparam (rejected ones too,
for the no-rejection sensitivity analysis), merging with the rejection flags."""
import numpy as np, pandas as pd, json, sys, warnings
from pathlib import Path
from scipy import signal as sig
warnings.filterwarnings("ignore")

# analyze_band, alpha_power, published_epoch_ri, BAND_SEARCH, ALPHA already in kernel
from specparam import SpectralModel

def specparam_alpha(x, fs):
    """specparam fit on the locked Welch PSD (3-40 Hz); returns alpha-band peak params."""
    import rhythmicity_locked as L1
    f, P, _ = L1.welch_psd(x, fs=fs)
    m = (f >= 3.0) & (f <= 40.0)
    sm = SpectralModel(peak_width_limits=(1.0, 8.0), max_n_peaks=6, verbose=False)
    try:
        sm.fit(f[m], P[m])
        ap = np.asarray(sm.results.params.periodic.params, float).reshape(-1, 3)
        pi = sm.results.params.periodic.indices
        met = sm.results.metrics.results
        r2 = float(met.get("gof_rsquared", np.nan))
        if len(ap):
            ap = ap[:, [pi["cf"], pi["pw"], pi["bw"]]]
            inb = ap[(ap[:, 0] >= ALPHA[0]) & (ap[:, 0] <= ALPHA[1])]
            n_alpha = len(inb)
            if n_alpha:
                k = int(np.argmax(inb[:, 1]))
                return dict(sp_ok=True, sp_r2=r2, sp_n_peaks=len(ap), sp_alpha_peaks=n_alpha,
                            sp_alpha_cf=float(inb[k, 0]), sp_alpha_pw=float(inb[k, 1]),
                            sp_alpha_bw=float(inb[k, 2]))
        return dict(sp_ok=True, sp_r2=r2, sp_n_peaks=len(ap), sp_alpha_peaks=0,
                    sp_alpha_cf=np.nan, sp_alpha_pw=0.0, sp_alpha_bw=np.nan)
    except Exception as e:
        return dict(sp_ok=False, sp_reason=str(e))

rows = []
for (subj, cond), st in store.items():
    fs = st["fs"]; elen = int(4.0 * fs)
    for e in range(len(st["bb"]) // elen):
        sl = slice(e * elen, (e + 1) * elen)
        x = st["bb"][sl] * 1e6
        row = dict(subject=subj, condition=cond, epoch=e)
        a1 = analyze_band(x, fs, BAND_SEARCH)
        for k_ in ("RI", "label", "admissible", "Q", "f0", "c", "cbar", "r",
                   "harmonic_k", "freq_precision_ok", "Q_resolution_limited",
                   "Q_undersegmented"):
            row[f"{k_}_v1"] = a1.get(k_)
        row["alpha_power_v1"] = alpha_power(x, fs)
        a2 = analyze_band(st["hp"][sl] * 1e6, fs, BAND_SEARCH)
        row.update(RI_v2=a2["RI"], admissible_v2=a2["admissible"], f0_v2=a2["f0"])
        a3 = analyze_band(st["nb"][sl] * 1e6, fs, ALPHA)
        row.update(RI_v3=a3["RI"], admissible_v3=a3["admissible"], f0_v3=a3["f0"])
        row["RI_v0"] = published_epoch_ri(st["bb"][sl], fs)
        row.update(specparam_alpha(x, fs))
        rows.append(row)

allep = pd.DataFrame(rows)
inv2 = inv[["subject", "condition", "epoch", "n_blinks", "ptp", "ptp_thr",
            "rej_blink", "rej_ptp", "rej_flat", "rejected"]]
allep = allep.merge(inv2, on=["subject", "condition", "epoch"], validate="1:1")
allep.to_csv("s4_per_epoch_full.csv", index=False)
print("rows:", len(allep), "| specparam ok:", int(allep.sp_ok.sum()),
      "| median sp_r2:", round(float(allep.sp_r2.median()), 3))
print("admissible v1 overall:", int(allep.admissible_v1.sum()))
print(allep.groupby("condition")[["RI_v1", "RI_v0"]].mean().round(3).to_string())
