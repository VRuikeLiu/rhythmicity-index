"""Session-2 extension of the locked Session-1 estimator.

`locked_s1.py` is the canonical specification for the revision and is NOT modified
here: Q is computed by its `q_factor` at the locked Welch setting (Hann, nperseg=256,
50% overlap, detrend='constant', -3 dB full width with interpolated crossings), and
the index formula, thresholds and admissibility floors are imported unchanged.

This module adds one thing, which the Session-1 verification pass identified as a
prerequisite for Session 2:

    The ANALYSIS FREQUENCY handed to the coherence term must be resolved well enough
    for the record length, and the Welch grid used for Q is not always fine enough.

Why it matters. The lagged-coherence term compares the phase of successive windows at
the analysis frequency f0. An error df in f0 rotates the phasor of each successive
window by a constant increment 2*pi*df*T_window, so the mean resultant length follows a
Dirichlet kernel in (df * record length) and collapses once the accumulated phase error
approaches a cycle. The tolerance therefore tightens as the record gets longer: holding
c >= 0.9 needs a relative frequency error below roughly 3% over 10 cycles but below
0.13% over 200. The Welch grid at nperseg = 256 has bin width fs/256, which for a
50-sample period puts the fundamental at bin 5 where one bin is 20% of f0 — sub-bin
parabolic interpolation on five coarse points cannot recover 0.1% from that. The
symptom, demonstrated in the Session-1 verification: an exactly periodic 120-sample
signal scores c = 0.034 and RI = 0.000 (ARRHYTHMIC) while passing the admissibility
gate, because the gate checks samples-per-cycle and cycle count but not whether the
frequency itself is resolved.

The fix, in three stages, all on the same measurement window:

  1. Coarse localisation. Take the dominant peak of the locked Welch PSD, which is what
     defines Q, and use it only to bracket the search.
  2. Fine localisation. Recompute a zero-padded periodogram of the whole (Hann-tapered,
     mean-removed) window at `PAD_FACTOR` times the natural resolution and take the
     largest local maximum inside the bracket, refined by parabolic interpolation on
     the log-magnitude. Zero padding does not create resolution, but it does remove the
     grid-quantisation error in locating a peak the record is already long enough to
     resolve -- which is exactly the error that collapses the coherence term.
  3. Harmonic correction. The Session-1 ACF-verified subharmonic search is applied to
     the refined frequency, unchanged in rule (accept f0/k only if its autocorrelation
     exceeds the incumbent by `margin` AND sits at a local ACF maximum).

A `freq_precision_ok` diagnostic is reported with every result: the estimated relative
frequency uncertainty (half the fine-grid spacing over f0, propagated through the
Dirichlet criterion) against the tolerance implied by the number of cycles in the
record. It is reported, not used to reject: a signal whose frequency is genuinely
unresolvable at the record length has an uninterpretable coherence value and that fact
belongs in the output table.

Nothing above changes the index definition, the thresholds, the admissibility floors,
or the Q convention. It changes only which frequency the components are measured at,
and it moves values only in the direction of the correct answer -- verified on exactly
periodic synthetics, where the Session-1 default returns c = 0.034 at a 120-sample
period and this module returns c = 1.000.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sig

import rhythmicity_locked as L1
from rhythmicity_locked import (T_C, T_CBAR, T_R, T_CW, T_CBARW, T_RW, T_MIN,
                        MIN_SAMPLES_PER_CYCLE, MIN_CYCLES, NPERSEG_DEFAULT,
                        index_gates, rhythmicity_index, classify, q_factor)

PAD_FACTOR = 8          # zero-padding multiple for the fine periodogram
SUBHARM_MAX = 8         # k range of the subharmonic search (unchanged from S1)
SUBHARM_MARGIN = 0.05   # ACF margin for accepting a subharmonic (unchanged from S1)


def fine_spectrum(x, fs=1.0, pad_factor=PAD_FACTOR):
    """Zero-padded Hann-tapered periodogram of the whole window (mean removed)."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    w = sig.get_window("hann", n)
    nfft = int(2 ** np.ceil(np.log2(max(8, n * pad_factor))))
    X = np.fft.rfft(x * w, n=nfft)
    f = np.fft.rfftfreq(nfft, d=1.0 / fs)
    P = (np.abs(X) ** 2) / (np.sum(w ** 2) * fs)
    return f[1:], P[1:], nfft          # DC dropped, matching the locked convention


def refine_peak_freq_fine(x, fs=1.0, nperseg=NPERSEG_DEFAULT, pad_factor=PAD_FACTOR):
    """Analysis frequency: Welch peak for bracketing, fine periodogram for location.

    Returns (f0, f_resolution) where f_resolution is the fine-grid spacing, i.e. the
    scale of the residual localisation error.
    """
    x = np.asarray(x, dtype=float)
    fw, Pw, nw = L1.welch_psd(x, fs=fs, nperseg=nperseg)
    if len(fw) < 3:
        return np.nan, np.nan
    pk, _ = sig.find_peaks(Pw)
    if len(pk) == 0:
        return np.nan, np.nan
    i = int(pk[np.argmax(Pw[pk])])
    df_welch = fs / nw
    lo, hi = fw[i] - 1.5 * df_welch, fw[i] + 1.5 * df_welch

    ff, Pf, nfft = fine_spectrum(x, fs=fs, pad_factor=pad_factor)
    df_fine = fs / nfft
    band = (ff >= lo) & (ff <= hi)
    if band.sum() < 3:
        return float(fw[i]), df_fine
    idx = np.flatnonzero(band)
    j = int(idx[np.argmax(Pf[idx])])
    if j <= 0 or j >= len(Pf) - 1:
        return float(ff[j]), df_fine
    y0, y1, y2 = np.log(Pf[j - 1] + 1e-300), np.log(Pf[j] + 1e-300), np.log(Pf[j + 1] + 1e-300)
    den = y0 - 2 * y1 + y2
    d = 0.0 if abs(den) < 1e-15 else float(np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5))
    return float(ff[j] + d * df_fine), df_fine


def fundamental_freq_fine(x, fs=1.0, nperseg=NPERSEG_DEFAULT, pad_factor=PAD_FACTOR,
                          max_sub=SUBHARM_MAX, margin=SUBHARM_MARGIN):
    """S1's ACF-verified subharmonic search applied to the finely-located peak."""
    fp, df_fine = refine_peak_freq_fine(x, fs=fs, nperseg=nperseg, pad_factor=pad_factor)
    if not np.isfinite(fp) or fp <= 0:
        return np.nan, 1, np.nan
    best_f, best_k = fp, 1
    best_a = L1._acf_at(x, max(1, int(round(fs / fp))))
    for k in range(2, max_sub + 1):
        Lk = int(round(k * fs / fp))
        if Lk >= len(x) // 3 or Lk < 2:
            break
        a = L1._acf_at(x, Lk)
        if a > best_a + margin and a >= L1._acf_at(x, Lk - 1) and a >= L1._acf_at(x, Lk + 1):
            best_f, best_k, best_a = fp / k, k, a
    return best_f, best_k, df_fine


def peak_power_share(x, fs=1.0, nperseg=NPERSEG_DEFAULT):
    """Fraction of total PSD power in the dominant peak's half-power band."""
    f, P, n = L1.welch_psd(x, fs=fs, nperseg=nperseg)
    if len(f) < 3 or P.sum() <= 0:
        return np.nan
    pk, _ = sig.find_peaks(P)
    if len(pk) == 0:
        return np.nan
    i = int(pk[np.argmax(P[pk])])
    half = P[i] / 2.0
    lo = i
    while lo > 0 and P[lo] > half:
        lo -= 1
    hi = i
    while hi < len(P) - 1 and P[hi] > half:
        hi += 1
    return float(P[lo:hi + 1].sum() / P.sum())


def analyze_s2(x, fs=1.0, nperseg=NPERSEG_DEFAULT,
               min_samples_per_cycle=MIN_SAMPLES_PER_CYCLE, min_cycles=MIN_CYCLES,
               pad_factor=PAD_FACTOR):
    """Locked pipeline with the S2 analysis-frequency refinement.

    Q, the index formula, the thresholds and the admissibility floors are exactly
    Session 1's. Only the frequency at which the components are measured differs.
    """
    from rhythmicity import (compute_lagged_coherence, compute_lagged_coherence_curve,
                              compute_cycle_consistency)
    x = np.asarray(x, dtype=float)
    q = q_factor(x, fs=fs, nperseg=nperseg)
    out = dict(Q=q["Q"], Q_peak_freq=q["peak_freq"], Q_bandwidth=q["bandwidth"],
               Q_bins_across=q["bins_across"], Q_n_segments=q["n_segments"],
               Q_resolution_limited=q["resolution_limited"],
               Q_undersegmented=q["undersegmented"],
               Q_ceiling=q.get("Q_ceiling", np.nan),
               peak_power_share=peak_power_share(x, fs=fs, nperseg=nperseg))

    if np.std(x) <= 0:
        out.update(RI=np.nan, label="UNDEFINED", admissible=False, reason="constant signal")
        return out

    f0, k, df_fine = fundamental_freq_fine(x, fs=fs, nperseg=nperseg, pad_factor=pad_factor)
    out.update(f0=f0, harmonic_k=k, f_resolution=df_fine)
    if not np.isfinite(f0) or f0 <= 0:
        out.update(RI=np.nan, label="UNDEFINED", admissible=False, reason="no spectral peak")
        return out

    P = fs / f0
    spc, ncyc = P, len(x) / P
    # Dirichlet criterion: phase error across the record stays below a quarter cycle.
    rel_unc = (0.5 * df_fine) / f0 if f0 > 0 else np.inf
    tol = 0.25 / max(ncyc, 1e-9)
    out.update(freq_rel_uncertainty=rel_unc, freq_rel_tolerance=tol,
               freq_precision_ok=bool(rel_unc <= tol))

    c = compute_lagged_coherence(x, fs, f0, n_cycles=3)
    cbar = compute_lagged_coherence_curve(x, fs, f0, max_n_cycles=10)["mean_coherence"]
    r = compute_cycle_consistency(x, P)["mean_corr"]
    ri = rhythmicity_index(c, cbar, r)

    reasons = []
    if spc < min_samples_per_cycle:
        reasons.append("samples/cycle %.2f < %d" % (spc, min_samples_per_cycle))
    if ncyc < min_cycles:
        reasons.append("cycles %.1f < %d" % (ncyc, min_cycles))
    out.update(period=P, samples_per_cycle=spc, n_cycles=ncyc, c=c, cbar=cbar, r=r,
               RI=ri, label=classify(ri), admissible=not reasons,
               reason="; ".join(reasons), **index_gates(c, cbar, r))
    return out
