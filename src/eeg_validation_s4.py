"""Session 4 pass 2 — final rejection design + all variants + baselines.

Rejection (MF-13), decided ONCE on broadband data, condition-blind, applied to all
variants and all baseline metrics (identical epochs everywhere):
  R1 blink: >=1 detected blink inside the epoch on the frontal mean (Fp1/Fpz/Fp2,
     |x| > 200 uV peaks, min separation 0.25 s). Dataset has no EOG channel; frontal
     poles are the standard proxy. Directly answers the reviewer's blink-asymmetry
     worry: contaminated epochs are excluded rather than statistically absorbed.
  R2 occipital PTP outlier: max channel PTP > subject's median + 4*1.4826*MAD over
     the subject's own 30 epochs (both conditions pooled, condition-blind). A fixed
     threshold is wrong here: occipital alpha itself is high-amplitude, so a fixed
     150 uV cut rejects MORE eyes-closed than eyes-open epochs (measured: 203 vs 149)
     and would bias against the very signal under test. Subject-calibrated MAD keeps
     the criterion amplitude-scale-free.
  R3 flat: PTP < 0.5 uV.
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import signal as sig

sys.path.insert(0, str(Path("repo/src").resolve()))
exec(open("s4_eeg_pipeline.py").read().split("# ---------------------------------------------------------------------------- load/loop")[0].split('"""', 2)[2])  # reuse imports + analyze_band etc.

import mne
from mne.datasets import eegbci
from scipy.signal import find_peaks as _fp
mne.set_log_level("ERROR"); warnings.filterwarnings("ignore")

DATA_DIR = Path("eeg_data"); OCC = ["O1", "Oz", "O2"]; FRONT = ["Fp1", "Fpz", "Fp2"]
EPOCH_S = 4.0; BLINK_UV = 200e-6; K_MAD = 4.0; FLAT = 0.5e-6

def load(subj, run):
    raw = mne.io.read_raw_edf(DATA_DIR / f"S{subj:03d}R{run:02d}.edf", preload=True)
    eegbci.standardize(raw); return raw

# ---- pass 1: epoch inventory (broadband) ----
inv = []
store = {}
for subj in range(1, 21):
    for cond, run in (("EO", 1), ("EC", 2)):
        raw = load(subj, run); fs = raw.info["sfreq"]
        occ = raw.copy().pick(OCC)
        fr = raw.copy().pick([c for c in FRONT if c in raw.ch_names])
        occ_bb = occ.copy().filter(1., 45., fir_design="firwin", verbose="ERROR")
        occ_hp = occ.copy().filter(1., None, fir_design="firwin", verbose="ERROR")
        occ_nb = occ.copy().filter(8., 13., fir_design="firwin", verbose="ERROR")
        fr_bb = fr.copy().filter(1., 45., fir_design="firwin", verbose="ERROR")
        sig_bb = occ_bb.get_data().mean(axis=0)
        store[(subj, cond)] = dict(bb=sig_bb, hp=occ_hp.get_data().mean(axis=0),
                                   nb=occ_nb.get_data().mean(axis=0),
                                   occ_ch=occ_bb.get_data(), fs=fs)
        fm = np.abs(fr_bb.get_data().mean(axis=0))
        pk, _ = _fp(fm, height=BLINK_UV, distance=int(0.25 * fs))
        elen = int(EPOCH_S * fs)
        for e in range(len(sig_bb) // elen):
            sl = slice(e * elen, (e + 1) * elen)
            inv.append(dict(subject=subj, condition=cond, epoch=e,
                            ptp=float(np.ptp(store[(subj, cond)]["occ_ch"][:, sl], axis=1).max()),
                            n_blinks=int(((pk >= sl.start) & (pk < sl.stop)).sum())))
inv = pd.DataFrame(inv)
thr = inv.groupby("subject").ptp.apply(
    lambda s: s.median() + K_MAD * 1.4826 * np.median(np.abs(s - s.median())))
inv["ptp_thr"] = inv.subject.map(thr)
inv["rej_blink"] = inv.n_blinks > 0
inv["rej_ptp"] = inv.ptp > inv.ptp_thr
inv["rej_flat"] = inv.ptp < FLAT
inv["rejected"] = inv.rej_blink | inv.rej_ptp | inv.rej_flat

# ---- pass 2: score accepted epochs, all variants ----
rows = []
for (subj, cond), g in inv.groupby(["subject", "condition"]):
    st = store[(subj, cond)]; fs = st["fs"]; elen = int(EPOCH_S * fs)
    for _, rr in g.iterrows():
        e = int(rr.epoch); sl = slice(e * elen, (e + 1) * elen)
        row = dict(subject=subj, condition=cond, epoch=e, rejected=bool(rr.rejected),
                   rej_blink=bool(rr.rej_blink), rej_ptp=bool(rr.rej_ptp),
                   n_blinks=int(rr.n_blinks), ptp_uV=rr.ptp * 1e6)
        row["RI_v0"] = published_epoch_ri(st["bb"][sl], fs)
        if not rr.rejected:
            x = st["bb"][sl] * 1e6
            a1 = analyze_band(x, fs, BAND_SEARCH)
            for k_ in ("RI", "label", "admissible", "Q", "f0", "c", "cbar", "r",
                       "harmonic_k", "freq_precision_ok", "Q_resolution_limited",
                       "Q_undersegmented", "n_cycles", "samples_per_cycle"):
                row[f"{k_}_v1"] = a1.get(k_)
            row["alpha_power_v1"] = alpha_power(x, fs)
            a2 = analyze_band(st["hp"][sl] * 1e6, fs, BAND_SEARCH)
            row.update(RI_v2=a2["RI"], admissible_v2=a2["admissible"],
                       label_v2=a2["label"], f0_v2=a2["f0"])
            a3 = analyze_band(st["nb"][sl] * 1e6, fs, ALPHA)
            row.update(RI_v3=a3["RI"], admissible_v3=a3["admissible"],
                       label_v3=a3["label"], f0_v3=a3["f0"])
        rows.append(row)
ep = pd.DataFrame(rows)
ep.to_csv("s4_per_epoch.csv", index=False)

rejtab = (inv.groupby("condition").agg(total=("rejected", "size"),
          rejected=("rejected", "sum"), blink=("rej_blink", "sum"),
          ptp=("rej_ptp", "sum"), flat=("rej_flat", "sum")))
print(rejtab.to_string())
acc = ep[~ep.rejected]
print("\naccepted per condition:", acc.groupby("condition").size().to_dict())
print("accepted per subject/cond min-max:",
      acc.groupby(["subject", "condition"]).size().min(),
      acc.groupby(["subject", "condition"]).size().max())
print("V1 admissible among accepted:", int(acc.admissible_v1.fillna(False).sum()), "of", len(acc))
print("V1 freq_precision_ok:", int(acc.freq_precision_ok_v1.fillna(False).sum()))
print("V0 check vs published: per-cond mean of RI_v0 on ALL epochs:")
print(ep.groupby("condition").RI_v0.mean().round(3).to_string())
