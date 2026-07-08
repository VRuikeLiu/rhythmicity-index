"""Validate the rhythmicity index on real EEG (Berger effect).

Applies the SAME rhythmicity index used on the simulation to human EEG from the
EEG Motor Movement/Imagery Database (EEGMMIDB / "EEGBCI"), PhysioNet
(Schalk et al. 2004; Goldberger et al. 2000):
    https://physionet.org/content/eegmmidb/1.0.0/
Baseline runs R01 (eyes-open) and R02 (eyes-closed) are analysed on occipital
channels. If the index is a valid periodicity measure it should be higher for
eyes-closed (alpha rhythm present) than eyes-open, across subjects.

Data downloads automatically via mne.datasets.eegbci (no EDF files committed).
Writes ../data/eeg_rhythmicity.csv (per subject x condition mean RI).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import welch, find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rhythmicity import classify_rhythmicity

N_SUBJECTS = 20
OCCIPITAL = ["Oz", "O1", "O2"]
ALPHA_BAND = (8.0, 13.0)
EPOCH_SECONDS = 4.0
OUT_CSV = Path(__file__).resolve().parent.parent / "data" / "eeg_rhythmicity.csv"


def _alpha_peak(signal, fs):
    """Return (peak_freq, has_alpha) from the Welch PSD in the alpha band."""
    freqs, psd = welch(signal - np.mean(signal), fs=fs,
                       nperseg=min(256, len(signal)), noverlap=min(128, len(signal) // 2))
    band = (freqs >= ALPHA_BAND[0]) & (freqs <= ALPHA_BAND[1])
    if not band.any():
        return 0.0, False
    af, ap = freqs[band], psd[band]
    i = int(np.argmax(ap))
    has_alpha = ap[i] > 2.0 * ap.mean() and ap[i] > np.median(psd) * 1.5
    return float(af[i]), bool(has_alpha)


def analyse_epoch(signal, fs):
    pf, has_alpha = _alpha_peak(signal, fs)
    if not has_alpha:
        return 0.0
    res = classify_rhythmicity(signal=np.asarray(signal, float), fs=fs,
                               peak_freq=pf, period=fs / pf, has_spectral_peak=True)
    return float(res["rhythmicity_index"])


def run(n_subjects: int = N_SUBJECTS) -> pd.DataFrame:
    import mne
    from mne.datasets import eegbci
    mne.set_log_level("ERROR")
    rows = []
    for subj in range(1, n_subjects + 1):
        for cond, runs in (("EO", [1]), ("EC", [2])):
            try:
                fpath = eegbci.load_data(subj, runs=runs, update_path=True, verbose=False)[0]
                raw = mne.io.read_raw_edf(fpath, preload=True, verbose=False)
                eegbci.standardize(raw)
                chans = [c for c in OCCIPITAL if c in raw.ch_names]
                if not chans:
                    continue
                raw.pick(chans).filter(1.0, 45.0, fir_design="firwin", verbose=False)
                fs = raw.info["sfreq"]
                data = raw.get_data().mean(axis=0)
                elen = int(EPOCH_SECONDS * fs)
                for s in range(0, len(data) - elen + 1, elen):
                    rows.append({"subject": subj, "condition": cond,
                                 "rhythmicity_index": analyse_epoch(data[s:s + elen], fs)})
            except Exception as e:  # pragma: no cover
                print(f"  subject {subj} {cond}: skipped ({e})")
    epochs = pd.DataFrame(rows)
    summary = (epochs.groupby(["subject", "condition"])
               .agg(mean_RI=("rhythmicity_index", "mean"),
                    n_epochs=("rhythmicity_index", "size")).reset_index())
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_CSV, index=False)
    ec, eo = summary[summary.condition == "EC"].mean_RI, summary[summary.condition == "EO"].mean_RI
    print(f"wrote {OUT_CSV}  |  EC mean RI = {ec.mean():.3f}, EO mean RI = {eo.mean():.3f}")
    return summary


if __name__ == "__main__":
    run()
