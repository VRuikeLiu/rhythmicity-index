"""Session 4 — EEG re-analysis under the locked estimator (MF-12..17, MF-24, REC-14, MIN-13/14).

Pipeline (all decisions documented in the grand log):
- Data: EEGMMIDB (PhysioNet), subjects 1-20, R01 eyes-open / R02 eyes-closed, fs=160 Hz.
- Channels: occipital O1, Oz, O2 (averaged into one series AFTER filtering — MIN-14);
  frontal Fp1, Fpz, Fp2 used only for blink detection (dataset has no EOG).
- Epochs: contiguous non-overlapping 4-s epochs from t=0 (matches the published grid).
- Artifact rejection (MF-13): decided ONCE on the broadband (1-45 Hz) copy, applied to
  every variant, so all variants and all baseline metrics see IDENTICAL epochs:
    reject if occipital PTP > 150 uV, or frontal PTP > 150 uV (blink proxy),
    or occipital PTP < 0.5 uV (flat).
- Variants (MF-12):
    V0  exact replication of the published pipeline (old rhythmicity.py, alpha-screen
        gate that assigns RI=0 to epochs without an alpha peak, no rejection) —
        reconciliation against the published per-subject table.
    V1  PRIMARY: broadband 1-45 Hz, free peak estimation in 3-20 Hz, locked S1 spec +
        S2 fine-frequency refinement, artifact rejection ON.
    V2  minimal filter: 1 Hz high-pass only.
    V3  narrowband 8-13 Hz filter (the reviewer's circularity condition).
- Baselines on V1's identical epochs (MF-17): band-limited locked Q, log10 alpha power
  (8-13 Hz), mean cycle correlation r, RI, specparam alpha peak power (MF-24).

Outputs: s4_per_epoch.csv, s4_per_subject.csv, s4_stats.json
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import signal as sig
from scipy import stats

sys.path.insert(0, str(Path("repo/src").resolve()))
import rhythmicity_locked as L1
import rhythmicity_locked_freq as F2
from rhythmicity import (compute_lagged_coherence, compute_lagged_coherence_curve,
                         compute_cycle_consistency, classify_rhythmicity)

import mne
from mne.datasets import eegbci
mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

DATA_DIR = Path("eeg_data")
OCC = ["O1", "Oz", "O2"]
FRONT = ["Fp1", "Fpz", "Fp2"]
EPOCH_S = 4.0
BAND_SEARCH = (3.0, 20.0)      # free peak-estimation band (V1/V2): admissible range at
                               # 4 s/160 Hz is [2.5, 20] Hz (>=10 cycles, >=8 samples/cycle)
ALPHA = (8.0, 13.0)
PTP_EEG = 150e-6               # V, occipital rejection
PTP_BLINK = 150e-6             # V, frontal (blink proxy)
FLAT = 0.5e-6

# ---------------------------------------------------------------- locked, band-limited
def q_from_peak(f, P, i, fs, n, K):
    """Locked -3 dB interpolated full width around peak index i (verbatim walk from
    rhythmicity_locked.q_factor; only the peak SELECTION differs upstream)."""
    df = fs / n
    half = P[i] / 2.0
    lo = i
    while lo > 0 and P[lo] > half:
        lo -= 1
    f_lo = f[lo] if P[lo] > half else f[lo] + (half - P[lo]) / (P[lo + 1] - P[lo]) * (f[lo + 1] - f[lo])
    hi = i
    while hi < len(P) - 1 and P[hi] > half:
        hi += 1
    f_hi = f[hi] if P[hi] > half else f[hi - 1] + (half - P[hi - 1]) / (P[hi] - P[hi - 1]) * (f[hi] - f[hi - 1])
    bw, f0 = float(f_hi - f_lo), float(f[i])
    return dict(Q=float(f0 / bw) if bw > 0 else np.nan, peak_freq=f0, bandwidth=bw,
                bins_across=bw / df, n_segments=K,
                resolution_limited=bool(bw / df < L1.MIN_PEAK_BINS),
                undersegmented=bool(K < L1.MIN_SEGMENTS))

def analyze_band(x, fs, band, nperseg=L1.NPERSEG_DEFAULT):
    """Locked S1 pipeline + S2 fine-frequency refinement, with the dominant-peak
    SELECTION restricted to `band`. Everything else (width walk, refinement bracket,
    ACF-verified subharmonic search, components, index, floors) is the locked code."""
    x = np.asarray(x, float)
    out = dict(RI=np.nan, label="UNDEFINED", admissible=False, reason="",
               Q=np.nan, f0=np.nan, c=np.nan, cbar=np.nan, r=np.nan,
               harmonic_k=1, freq_precision_ok=False,
               Q_resolution_limited=True, Q_undersegmented=True)
    if np.std(x) <= 0:
        out["reason"] = "constant signal"; return out
    fw, Pw, n = L1.welch_psd(x, fs=fs, nperseg=nperseg)
    K = 1 + max(0, (len(x) - n)) // max(1, n // 2)
    pk, _ = sig.find_peaks(Pw)
    pk = pk[(fw[pk] >= band[0]) & (fw[pk] <= band[1])]
    if len(pk) == 0:
        out["reason"] = "no spectral peak in band"; return out
    i = int(pk[np.argmax(Pw[pk])])
    q = q_from_peak(fw, Pw, i, fs, n, K)
    out.update(Q=q["Q"], Q_resolution_limited=q["resolution_limited"],
               Q_undersegmented=q["undersegmented"], Q_peak_freq=q["peak_freq"],
               Q_bins_across=q["bins_across"])
    # S2 fine refinement, bracketed on the band-selected Welch peak
    df_welch = fs / n
    ff, Pf, nfft = F2.fine_spectrum(x, fs=fs)
    df_fine = fs / nfft
    m = (ff >= fw[i] - 1.5 * df_welch) & (ff <= fw[i] + 1.5 * df_welch)
    if m.sum() >= 3:
        idx = np.flatnonzero(m)
        j = int(idx[np.argmax(Pf[idx])])
        if 0 < j < len(Pf) - 1:
            y0, y1, y2 = np.log(Pf[j-1] + 1e-300), np.log(Pf[j] + 1e-300), np.log(Pf[j+1] + 1e-300)
            den = y0 - 2 * y1 + y2
            d = 0.0 if abs(den) < 1e-15 else float(np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5))
            fp = float(ff[j] + d * df_fine)
        else:
            fp = float(ff[j])
    else:
        fp = float(fw[i])
    # ACF-verified subharmonic search (locked rule, unchanged)
    best_f, best_k = fp, 1
    best_a = L1._acf_at(x, max(1, int(round(fs / fp))))
    for k in range(2, 9):
        Lk = int(round(k * fs / fp))
        if Lk >= len(x) // 3 or Lk < 2:
            break
        a = L1._acf_at(x, Lk)
        if a > best_a + 0.05 and a >= L1._acf_at(x, Lk - 1) and a >= L1._acf_at(x, Lk + 1):
            best_f, best_k, best_a = fp / k, k, a
    f0 = best_f
    P = fs / f0
    spc, ncyc = P, len(x) / P
    rel_unc = (0.5 * df_fine) / f0
    tol = 0.25 / max(ncyc, 1e-9)
    c = compute_lagged_coherence(x, fs, f0, n_cycles=3)
    cbar = compute_lagged_coherence_curve(x, fs, f0, max_n_cycles=10)["mean_coherence"]
    r = compute_cycle_consistency(x, P)["mean_corr"]
    ri = L1.rhythmicity_index(c, cbar, r)
    reasons = []
    if spc < L1.MIN_SAMPLES_PER_CYCLE:
        reasons.append("samples/cycle %.2f < %d" % (spc, L1.MIN_SAMPLES_PER_CYCLE))
    if ncyc < L1.MIN_CYCLES:
        reasons.append("cycles %.1f < %d" % (ncyc, L1.MIN_CYCLES))
    out.update(f0=f0, harmonic_k=best_k, period=P, samples_per_cycle=spc, n_cycles=ncyc,
               c=c, cbar=cbar, r=r, RI=ri, label=L1.classify(ri),
               admissible=not reasons, reason="; ".join(reasons),
               freq_precision_ok=bool(rel_unc <= tol))
    return out

def alpha_power(x, fs):
    f, P, _ = L1.welch_psd(x, fs=fs)
    m = (f >= ALPHA[0]) & (f <= ALPHA[1])
    return float(np.trapezoid(P[m], f[m]))

def published_epoch_ri(x, fs):
    """V0: verbatim re-implementation of the published analyse_epoch (eeg_validation.py)."""
    f, P = sig.welch(x - np.mean(x), fs=fs, nperseg=min(256, len(x)),
                     noverlap=min(128, len(x) // 2))
    band = (f >= ALPHA[0]) & (f <= ALPHA[1])
    if not band.any():
        return 0.0
    af, ap = f[band], P[band]
    i = int(np.argmax(ap))
    has_alpha = ap[i] > 2.0 * ap.mean() and ap[i] > np.median(P) * 1.5
    if not has_alpha:
        return 0.0
    res = classify_rhythmicity(signal=np.asarray(x, float), fs=fs, peak_freq=float(af[i]),
                               period=fs / float(af[i]), has_spectral_peak=True)
    return float(res["rhythmicity_index"])

# ---------------------------------------------------------------------------- load/loop
def load_subject(subj, run):
    raw = mne.io.read_raw_edf(DATA_DIR / f"S{subj:03d}R{run:02d}.edf", preload=True)
    eegbci.standardize(raw)
    return raw

filter_doc = {}
rows = []
rej_rows = []
for subj in range(1, 21):
    for cond, run in (("EO", 1), ("EC", 2)):
        raw = load_subject(subj, run)
        fs = raw.info["sfreq"]
        occ = raw.copy().pick(OCC)
        front = raw.copy().pick([c for c in FRONT if c in raw.ch_names])
        # filtered copies (zero-phase FIR firwin; parameters recorded once)
        occ_bb = occ.copy().filter(1.0, 45.0, fir_design="firwin", verbose="ERROR")
        occ_hp = occ.copy().filter(1.0, None, fir_design="firwin", verbose="ERROR")
        occ_nb = occ.copy().filter(8.0, 13.0, fir_design="firwin", verbose="ERROR")
        front_bb = front.copy().filter(1.0, 45.0, fir_design="firwin", verbose="ERROR")
        if not filter_doc:
            for name, lf, hf in (("broadband", 1.0, 45.0), ("highpass", 1.0, None),
                                 ("narrowband", 8.0, 13.0)):
                h = mne.filter.create_filter(occ.get_data(), fs, lf, hf,
                                             fir_design="firwin", verbose="ERROR")
                filter_doc[name] = dict(l_freq=lf, h_freq=hf, n_taps=len(h),
                                        order=len(h) - 1, dur_s=len(h) / fs)
        sig_bb = occ_bb.get_data().mean(axis=0)
        sig_hp = occ_hp.get_data().mean(axis=0)
        sig_nb = occ_nb.get_data().mean(axis=0)
        fr_bb = front_bb.get_data()
        occ_ch = occ_bb.get_data()
        elen = int(EPOCH_S * fs)
        n_ep = len(sig_bb) // elen
        for e in range(n_ep):
            sl = slice(e * elen, (e + 1) * elen)
            ptp_occ = float(np.ptp(occ_ch[:, sl], axis=1).max())
            ptp_fr = float(np.ptp(fr_bb[:, sl], axis=1).max()) if len(fr_bb) else 0.0
            reasons = []
            if ptp_occ > PTP_EEG: reasons.append("occ_ptp")
            if ptp_fr > PTP_BLINK: reasons.append("frontal_ptp")
            if ptp_occ < FLAT: reasons.append("flat")
            rej_rows.append(dict(subject=subj, condition=cond, epoch=e,
                                 ptp_occ_uV=ptp_occ * 1e6, ptp_frontal_uV=ptp_fr * 1e6,
                                 rejected=bool(reasons), reason="+".join(reasons)))
            row = dict(subject=subj, condition=cond, epoch=e, rejected=bool(reasons))
            # V0 replication (published pipeline, no rejection, on broadband copy)
            row["RI_v0"] = published_epoch_ri(sig_bb[sl], fs)
            if not reasons:
                x = sig_bb[sl] * 1e6  # uV, scale-invariant for all metrics
                a1 = analyze_band(x, fs, BAND_SEARCH)
                for k_ in ("RI", "label", "admissible", "Q", "f0", "c", "cbar", "r",
                           "harmonic_k", "freq_precision_ok", "Q_resolution_limited",
                           "Q_undersegmented", "n_cycles", "samples_per_cycle"):
                    row[f"{k_}_v1"] = a1.get(k_)
                row["alpha_power_v1"] = alpha_power(x, fs)
                a2 = analyze_band(sig_hp[sl] * 1e6, fs, BAND_SEARCH)
                row.update(RI_v2=a2["RI"], admissible_v2=a2["admissible"],
                           label_v2=a2["label"], f0_v2=a2["f0"])
                a3 = analyze_band(sig_nb[sl] * 1e6, fs, ALPHA)
                row.update(RI_v3=a3["RI"], admissible_v3=a3["admissible"],
                           label_v3=a3["label"], f0_v3=a3["f0"])
            rows.append(row)

ep = pd.DataFrame(rows)
rej = pd.DataFrame(rej_rows)
ep.to_csv("s4_per_epoch.csv", index=False)
rej.to_csv("s4_rejection_table.csv", index=False)
print("epochs total:", len(ep), "| rejected:", int(ep.rejected.sum()))
print(rej.groupby(["condition"]).rejected.agg(["sum", "size"]).to_string())
print(rej[rej.rejected].groupby(["condition", "reason"]).size().to_string())
print("filters:", json.dumps(filter_doc))
print("V1 admissible among accepted:", int(ep.admissible_v1.fillna(False).sum()),
      "of", int((~ep.rejected).sum()))
print("V1 harmonic_k>1:", int((ep.harmonic_k_v1.fillna(1) > 1).sum()),
      "| freq_precision_ok:", int(ep.freq_precision_ok_v1.fillna(False).sum()))
