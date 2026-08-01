"""Spectral (Q-factor) baseline — the comparison the rhythmicity index is measured against.

Implements Methods section 2.2 of the paper: estimate the power spectral density of
an activity time series with Welch's method, locate the dominant spectral peak, and
compute the quality factor

    Q = f_center / delta_f

the ratio of the peak's centre frequency to its half-power (-3 dB) bandwidth. Q is the
standard summary of spectral-peak *sharpness*, and it is the quantity the paper shows to
be insufficient as evidence of genuine periodicity: it responds to regular timing and is
essentially blind to whether the waveform repeats (see `rhythmicity.py`).

Convention note (matters for reproducing the paper's numbers)
------------------------------------------------------------
Q depends on the spectral resolution, so the estimator's settings are part of the
measurement. The paper uses `nperseg = min(256, len(signal))` and takes the half-power
bandwidth from the outermost PSD bins still above half the peak power, without
sub-bin interpolation. With those settings the two runs in Figure 3 / Table 1 give
Q = 63.5 (highest-Q run) and Q = 28.3 (highest-RI run). Changing `nperseg` changes Q
substantially — at nperseg=512 the same highest-Q trace reads Q = 84.7 — so the
default here is the paper's value and should be kept when comparing against it.

References
----------
Welch, P. (1967). The use of FFT for the estimation of power spectra. IEEE ASSP 15(2).
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sig
from scipy.signal import find_peaks

#: Welch segment length used throughout the paper (see convention note above).
NPERSEG_DEFAULT: int = 256


def power_spectrum(signal: np.ndarray, fs: float = 1.0,
                   nperseg: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD of `signal`, mean-removed, with the DC bin dropped.

    Returns
    -------
    (freqs, psd) : both 1-D, strictly positive frequencies only.
    """
    x = np.asarray(signal, dtype=float)
    x = x - x.mean()
    n = int(nperseg if nperseg is not None else NPERSEG_DEFAULT)
    freqs, psd = sig.welch(x, fs=fs, nperseg=min(n, len(x)))
    keep = freqs > 0
    return freqs[keep], psd[keep]


def compute_q_factor(signal: np.ndarray, fs: float = 1.0,
                     nperseg: int | None = None) -> dict:
    """Dominant spectral peak and its quality factor Q = f_center / delta_f.

    The peak is the largest local maximum of the Welch PSD. Its half-power bandwidth
    is taken between the outermost bins on either side that remain above half the peak
    power (no sub-bin interpolation — see the module docstring).

    Parameters
    ----------
    signal : 1-D array
        Activity time series, already restricted to the stationary measurement window.
    fs : float
        Sampling frequency. For simulated activity traces this is 1.0 (one sample per
        timestep), so frequency is in cycles/timestep and `period` is in timesteps.
    nperseg : int, optional
        Welch segment length. Defaults to the paper's 256.

    Returns
    -------
    dict with keys:
        'Q'           — quality factor (nan if the bandwidth is unresolved)
        'peak_freq'   — centre frequency of the dominant peak
        'period'      — 1 / peak_freq, in samples
        'bandwidth'   — half-power bandwidth delta_f
        'peak_power'  — PSD value at the peak
        'has_peak'    — False when no local maximum was found at all
    """
    freqs, psd = power_spectrum(signal, fs=fs, nperseg=nperseg)
    none = {"Q": np.nan, "peak_freq": np.nan, "period": np.nan,
            "bandwidth": np.nan, "peak_power": np.nan, "has_peak": False}
    if len(freqs) < 3:
        return none

    peaks, _ = find_peaks(psd)
    if len(peaks) == 0:
        return none

    i = int(peaks[np.argmax(psd[peaks])])
    half = psd[i] / 2.0

    lo = i
    while lo > 0 and psd[lo] > half:
        lo -= 1
    hi = i
    while hi < len(psd) - 1 and psd[hi] > half:
        hi += 1

    bandwidth = float(freqs[hi] - freqs[lo])
    f_center = float(freqs[i])
    return {
        "Q": float(f_center / bandwidth) if bandwidth > 0 else np.nan,
        "peak_freq": f_center,
        "period": float(1.0 / f_center) if f_center > 0 else np.nan,
        "bandwidth": bandwidth,
        "peak_power": float(psd[i]),
        "has_peak": True,
    }
