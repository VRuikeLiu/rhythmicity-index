"""Locked rhythmicity-index and Q-factor estimator (NHSJS revision, Session 1).

This module is the CANONICAL specification referenced by Methods 2.2 and 2.3 of the
revised manuscript. Every RI and Q value reported in the paper must come from here.

Changes relative to the as-submitted implementation, each traceable to a reviewer item:

  MF-8 / REC-12  The intermediate (weak) branch used ``min(1 + (g_w - 1), 1.999)``, a hard
                 cap that pinned 56.8% of weak-band signals to 1.999 -- which prints as
                 "2.00" at two decimals and so displays a non-rhythmic signal with the
                 rhythmic label. Replaced by the compressive map ``2 - 1/g_w``, which is
                 strictly increasing on [1, inf), maps g_w = 1 to exactly 1, and approaches
                 2 only asymptotically. The strong branch is rewritten as ``1 + g_s`` --
                 algebraically identical to ``2 + (g_s - 1)`` but with no fictitious offset.
                 The ``0.999`` cap in the third branch is dropped: the branch condition
                 already confines max(g_s, g_w) below 1, so the cap only clipped the narrow
                 interval (0.999, 1) -- reachable, but affecting 0.003% of random component
                 triples and 10 of the 8948 surviving cells in the published sweep.

  MF-10          Q is fully specified: Welch, Hann window, 50% overlap, detrend='constant',
                 density scaling; dominant peak = largest local maximum of the PSD;
                 Delta_f = the -3 dB (half-power) FULL width with linear sub-bin
                 interpolation of both crossings. Two diagnostics accompany every Q:
                 ``resolution_limited`` (peak spans fewer than ``min_bins`` FFT bins, so Q
                 is a resolution ceiling rather than a property of the signal) and
                 ``undersegmented`` (fewer than ``min_segments`` averaged Welch segments,
                 so the PSD is too noisy for a stable half-power width).

  MF-5           A minimum samples-per-cycle floor and a minimum-cycle-count floor are
                 imposed and reported. A mean-removed cycle of n samples spans n-1
                 dimensions, of which the fundamental occupies 2; the shape term therefore
                 has n-3 degrees of freedom and is vacuous at n = 3.

  (new, S1)      Two estimator defects found while locking the spec:
                 (a) the analysis frequency was the raw Welch bin centre, whose quantization
                     error rotates the per-window phasors and collapses the coherence term
                     -- an exactly periodic signal could score c = 0.089 instead of 1.000.
                     Fixed by sub-bin parabolic interpolation on the log-PSD.
                 (b) when a harmonic dominates the PSD, the period fed to the shape term was
                     a fraction of the true repetition period, destroying the cycle
                     correlation. Fixed by an autocorrelation-verified subharmonic search.

The three component measures (lagged coherence c and c-bar, cycle correlation r) and the
gate thresholds are UNCHANGED from the published index.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sig

# --- gate thresholds (unchanged from the published index) -------------------------
T_C, T_CBAR, T_R = 0.5, 0.35, 0.4          # strong thresholds
T_CW, T_CBARW, T_RW = 0.3, 0.25, 0.2       # looser thresholds
T_MIN = 0.1                                # minimal-evidence floor

# --- admissibility floors (Session 1, MF-5) ---------------------------------------
MIN_SAMPLES_PER_CYCLE = 8
MIN_CYCLES = 10

# --- Q diagnostics (Session 1, MF-10) ---------------------------------------------
MIN_PEAK_BINS = 3.0
MIN_SEGMENTS = 8
NPERSEG_DEFAULT = 256


def welch_psd(x, fs=1.0, nperseg=NPERSEG_DEFAULT, window="hann", detrend="constant"):
    """Welch PSD, DC bin dropped. 50% overlap; this is the locked convention."""
    x = np.asarray(x, dtype=float)
    n = int(min(nperseg, len(x)))
    f, P = sig.welch(x, fs=fs, window=window, nperseg=n, noverlap=n // 2,
                     detrend=detrend, scaling="density")
    keep = f > 0
    return f[keep], P[keep], n


def q_factor(x, fs=1.0, nperseg=NPERSEG_DEFAULT, min_bins=MIN_PEAK_BINS,
             min_segments=MIN_SEGMENTS, **kw):
    """Q = f0 / Delta_f with interpolated half-power width, plus validity diagnostics."""
    x = np.asarray(x, dtype=float)
    f, P, n = welch_psd(x, fs=fs, nperseg=nperseg, **kw)
    K = 1 + max(0, (len(x) - n)) // max(1, n // 2)
    df = fs / n
    out = dict(Q=np.nan, peak_freq=np.nan, period=np.nan, bandwidth=np.nan,
               bins_across=np.nan, n_segments=K, df_bin=df, has_peak=False,
               resolution_limited=True, undersegmented=bool(K < min_segments))
    if len(f) < 3:
        return out
    pk, _ = sig.find_peaks(P)
    if len(pk) == 0:
        return out
    i = int(pk[np.argmax(P[pk])])
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
    out.update(Q=float(f0 / bw) if bw > 0 else np.nan, peak_freq=f0,
               period=float(1.0 / f0) if f0 > 0 else np.nan, bandwidth=bw,
               bins_across=bw / df, has_peak=True,
               resolution_limited=bool(bw / df < min_bins),
               Q_ceiling=float(f0 / (min_bins * df)))
    return out


def refine_peak_freq(x, fs=1.0, nperseg=NPERSEG_DEFAULT, **kw):
    """Sub-bin peak frequency: parabolic interpolation on the log-PSD (MF-10, defect a)."""
    f, P, _ = welch_psd(x, fs=fs, nperseg=nperseg, **kw)
    pk, _ = sig.find_peaks(P)
    if len(pk) == 0:
        return np.nan
    i = int(pk[np.argmax(P[pk])])
    if i == 0 or i == len(P) - 1:
        return float(f[i])
    y0, y1, y2 = np.log(P[i - 1]), np.log(P[i]), np.log(P[i + 1])
    den = y0 - 2 * y1 + y2
    d = 0.0 if abs(den) < 1e-15 else float(np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5))
    return float(f[i] + d * (f[1] - f[0]))


def _acf_at(x, lag):
    x = np.asarray(x, dtype=float) - np.mean(x)
    if lag < 1 or lag >= len(x):
        return 0.0
    a, b = x[:-lag], x[lag:]
    sa, sb = a.std(), b.std()
    return float(np.mean(a * b) / (sa * sb)) if sa > 1e-12 and sb > 1e-12 else 0.0


def fundamental_freq(x, fs=1.0, nperseg=NPERSEG_DEFAULT, max_sub=8, margin=0.05, **kw):
    """Dominant peak corrected for harmonic misidentification (MF-10, defect b).

    A subharmonic f_p/k is accepted only if its autocorrelation EXCEEDS the incumbent
    by `margin` and sits at a local ACF maximum. Exact multiples of a true period tie
    rather than exceed, so ties are rejected; requiring a local maximum rejects the
    monotone ACF ramp that produces spurious detections on noise.
    """
    fp = refine_peak_freq(x, fs=fs, nperseg=nperseg, **kw)
    if not np.isfinite(fp) or fp <= 0:
        return np.nan, 1
    best_f, best_k = fp, 1
    best_a = _acf_at(x, max(1, int(round(fs / fp))))
    for k in range(2, max_sub + 1):
        L = int(round(k * fs / fp))
        if L >= len(x) // 3 or L < 2:
            break
        a = _acf_at(x, L)
        if a > best_a + margin and a >= _acf_at(x, L - 1) and a >= _acf_at(x, L + 1):
            best_f, best_k, best_a = fp / k, k, a
    return best_f, best_k


def index_gates(c, cbar, r):
    """Intermediate quantities of the index. Thresholds unchanged from the published form."""
    c, cbar, r = max(float(c), 0.0), max(float(cbar), 0.0), max(float(r), 0.0)
    Phi = max(c / T_C, cbar / T_CBAR)
    Psi = r / T_R
    Phi_w = max(c / T_CW, cbar / T_CBARW)
    Psi_w = r / T_RW
    Phi_0 = max(c, cbar) / T_MIN if max(c, cbar) > T_MIN else 0.0
    Psi_0 = r / T_MIN if r > T_MIN else 0.0
    gs = min(Phi, Psi)
    gw = max(min(Phi_w, Psi_0), min(Psi_w, Phi_0))
    return dict(Phi=Phi, Psi=Psi, Phi_w=Phi_w, Psi_w=Psi_w, Phi_0=Phi_0, Psi_0=Psi_0,
                g_s=gs, g_w=gw)


def rhythmicity_index(c, cbar, r):
    """RI in [0, 3.5]. Branch ranges are exactly [0,1), [1,2), [2,3.5] -- no caps (MF-8)."""
    g = index_gates(c, cbar, r)
    gs, gw = g["g_s"], g["g_w"]
    if gs >= 1.0:
        return float(1.0 + gs)                 # rhythmic:   [2, 3.5]
    if gw >= 1.0:
        return float(2.0 - 1.0 / gw)           # weak:       [1, 2)
    return float(max(gs, gw))                  # arrhythmic: [0, 1)


def classify(ri):
    if not np.isfinite(ri):
        return "UNDEFINED"
    return "RHYTHMIC" if ri >= 2.0 else ("WEAKLY_RHYTHMIC" if ri >= 1.0 else "ARRHYTHMIC")


def analyze(x, fs=1.0, nperseg=NPERSEG_DEFAULT,
            min_samples_per_cycle=MIN_SAMPLES_PER_CYCLE, min_cycles=MIN_CYCLES):
    """Full locked pipeline. `admissible` False means the RI must not be reported (MF-5)."""
    from rhythmicity import (compute_lagged_coherence, compute_lagged_coherence_curve,
                             compute_cycle_consistency)
    x = np.asarray(x, dtype=float)
    q = q_factor(x, fs=fs, nperseg=nperseg)
    f0, k = fundamental_freq(x, fs=fs, nperseg=nperseg)
    out = dict(Q=q["Q"], Q_peak_freq=q["peak_freq"], Q_bins_across=q["bins_across"],
               Q_n_segments=q["n_segments"], Q_resolution_limited=q["resolution_limited"],
               Q_undersegmented=q["undersegmented"], harmonic_k=k, f0=f0)
    if not np.isfinite(f0) or f0 <= 0:
        out.update(RI=np.nan, label="UNDEFINED", admissible=False, reason="no spectral peak")
        return out
    P = fs / f0
    spc, ncyc = P, len(x) / P
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
