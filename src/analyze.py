"""One-call entry point: score any 1-D signal with both Q and the rhythmicity index.

This is the convenience wrapper most users want. It performs the two measurements the
paper compares — the spectral Q-factor (`spectral.py`) and the rhythmicity index
(`rhythmicity.py`) — on the same signal over the same window, using the paper's
conventions for both.

    from src.analyze import analyze
    res = analyze(my_signal, fs=1.0)
    print(res["Q"], res["RI"], res["rhythmicity_class"])

The period handed to the rhythmicity index is taken from the dominant spectral peak,
which is how the paper does it: the spectral estimate supplies the candidate period, and
the rhythmicity index then tests — in the time domain — whether the waveform actually
repeats at that period. If no spectral peak is found, `rhythmicity.classify_rhythmicity`
falls back to an autocorrelation-based period estimate.
"""
from __future__ import annotations

import numpy as np

from spectral import compute_q_factor
from rhythmicity import classify_rhythmicity


def analyze(signal: np.ndarray, fs: float = 1.0,
            nperseg: int | None = None) -> dict:
    """Compute the Q-factor and the rhythmicity index for one signal.

    Parameters
    ----------
    signal : 1-D array
        The measurement window — for simulated runs, the post-burn-in stationary
        segment (the paper uses timesteps 500-2000 of a 2500-step run).
    fs : float
        Sampling frequency. 1.0 for simulated activity traces (frequency in
        cycles/timestep); 160.0 for the EEGMMIDB recordings.
    nperseg : int, optional
        Welch segment length; defaults to the paper's 256. Q is sensitive to this.

    Returns
    -------
    dict with the spectral quantities ('Q', 'peak_freq', 'period', 'bandwidth',
    'has_peak'), the rhythmicity quantities ('RI', 'rhythmicity_class',
    'lagged_coherence', 'mean_curve_coherence', 'mean_cycle_corr'), and
    'period_source' indicating whether the period came from the spectrum or the ACF
    fallback.
    """
    x = np.asarray(signal, dtype=float)
    spec = compute_q_factor(x, fs=fs, nperseg=nperseg)

    peak_freq = spec["peak_freq"] if spec["has_peak"] else 0.0
    period = (fs / peak_freq) if (spec["has_peak"] and peak_freq > 0) else 0.0
    q_for_call = spec["Q"] if np.isfinite(spec["Q"]) else 0.0

    rhy = classify_rhythmicity(
        signal=x, fs=fs,
        peak_freq=float(peak_freq),
        period=float(period),
        Q_factor=float(q_for_call),
        has_spectral_peak=bool(spec["has_peak"]),
    )

    curve = rhy.get("lagged_coherence_curve") or {}
    return {
        # spectral baseline
        "Q": spec["Q"],
        "peak_freq": spec["peak_freq"],
        "period": spec["period"],
        "bandwidth": spec["bandwidth"],
        "has_peak": spec["has_peak"],
        # rhythmicity index
        "RI": rhy["rhythmicity_index"],
        "rhythmicity_class": rhy["rhythmicity_class"],
        "lagged_coherence": rhy["lagged_coherence"],
        "mean_curve_coherence": curve.get("mean_coherence", np.nan),
        "mean_cycle_corr": rhy["mean_cycle_corr"],
        "period_source": rhy.get("period_source"),
    }
