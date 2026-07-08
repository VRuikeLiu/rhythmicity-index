"""Rhythmicity analysis: lagged coherence, cycle consistency, and classification.

Implements the professor's definition of oscillation: "cycles or patterns that
repeat themselves" — not just "things going up and down."

Metrics:
  - Lagged coherence (Fransen et al. 2015): phase consistency across windows
  - Cycle consistency (Cole & Voytek 2019 inspired): waveform correlation
    between successive cycles, using peak/trough-aligned extraction

Classification:
  RHYTHMIC         — genuine repeating pattern (high coherence + consistency)
  WEAKLY_RHYTHMIC  — some repetition but noisy (borderline)
  ARRHYTHMIC       — no repeating pattern ("things going up and down")

References:
  Fransen et al. 2015, "Identifying neuronal oscillations using rhythmicity"
  Cole & Voytek 2019, "Cycle-by-cycle analysis of neural oscillations"
  Donoghue et al. 2020, "Parameterizing neural power spectra" (FOOOF)
  Cho et al. 2024, "Cyclic Homogeneous Oscillation detection" (eLife)
"""

import numpy as np
from scipy import signal as sig


# ---------------------------------------------------------------------------
# Lagged coherence (unchanged)
# ---------------------------------------------------------------------------

def compute_lagged_coherence(signal: np.ndarray, fs: float, freq_of_interest: float,
                             n_cycles: int = 3) -> float:
    """Compute lagged coherence at a single lag (Fransen et al. 2015).

    Segments the signal into non-overlapping windows of `n_cycles` periods,
    computes the Fourier coefficient at `freq_of_interest` for each window,
    and returns the phase consistency (mean resultant length) across windows.

    Parameters
    ----------
    signal : 1D time series (measurement window, post-burn-in)
    fs : sampling frequency (1.0 for discrete timesteps)
    freq_of_interest : dominant frequency (Hz) — typically peak_freq
    n_cycles : number of cycles per window

    Returns
    -------
    coherence : float in [0, 1]. 0 = no phase consistency, 1 = perfect.
    """
    if freq_of_interest <= 0 or fs <= 0:
        return 0.0

    period = fs / freq_of_interest
    window_len = int(np.round(n_cycles * period))

    if window_len < 2:
        return 0.0

    n_windows = len(signal) // window_len
    if n_windows < 2:
        return 0.0

    # Compute Fourier coefficient at freq_of_interest for each window
    fourier_coeffs = np.empty(n_windows, dtype=np.complex128)
    t_win = np.arange(window_len) / fs
    for i in range(n_windows):
        segment = signal[i * window_len:(i + 1) * window_len].astype(np.float64)
        segment = segment - np.mean(segment)
        # Fourier coefficient at the frequency of interest
        fourier_coeffs[i] = np.sum(segment * np.exp(-2j * np.pi * freq_of_interest * t_win))

    # If all Fourier coefficients have negligible magnitude, the signal
    # has no power at this frequency (e.g. constant signal) -> coherence = 0
    magnitudes = np.abs(fourier_coeffs)
    if np.max(magnitudes) < 1e-10:
        return 0.0

    # Mean resultant length (circular statistics) using unit-normalized coefficients
    unit_vectors = fourier_coeffs / magnitudes.clip(min=1e-20)
    coherence = float(np.abs(np.mean(unit_vectors)))

    return coherence


def compute_lagged_coherence_curve(signal: np.ndarray, fs: float,
                                   freq_of_interest: float,
                                   max_n_cycles: int = 10) -> dict:
    """Compute lagged coherence at multiple lags (1-cycle through max_n_cycles).

    Shows how far into the future the pattern persists. A true oscillation
    maintains coherence across many lags; noise drops off quickly.

    Parameters
    ----------
    signal : 1D time series
    fs : sampling frequency
    freq_of_interest : dominant frequency
    max_n_cycles : maximum lag in cycles

    Returns
    -------
    dict with:
        'n_cycles' : list of lag values (in cycles)
        'coherence' : list of coherence values at each lag
        'mean_coherence' : mean across all computed lags
    """
    n_cycles_list = []
    coherence_list = []

    for nc in range(1, max_n_cycles + 1):
        coh = compute_lagged_coherence(signal, fs, freq_of_interest, n_cycles=nc)
        # Only include if we had enough windows
        period = fs / freq_of_interest if freq_of_interest > 0 else len(signal)
        window_len = int(np.round(nc * period))
        if window_len < 2 or len(signal) // window_len < 2:
            break
        n_cycles_list.append(nc)
        coherence_list.append(coh)

    if not coherence_list:
        return {
            "n_cycles": [],
            "coherence": [],
            "mean_coherence": 0.0,
        }

    return {
        "n_cycles": n_cycles_list,
        "coherence": coherence_list,
        "mean_coherence": float(np.mean(coherence_list)),
    }


# ---------------------------------------------------------------------------
# Cycle extraction — aligned (new) + rigid (preserved for backward compat)
# ---------------------------------------------------------------------------

def extract_cycles(signal: np.ndarray, period: float) -> list[np.ndarray]:
    """Extract individual cycles from the signal using rigid equal-length bins.

    Preserved for backward compatibility (used by rhythmicity_figures.py).

    Parameters
    ----------
    signal : 1D time series
    period : cycle length in samples

    Returns
    -------
    list of 1D arrays, each one cycle long
    """
    period_int = int(np.round(period))
    if period_int < 2:
        return []

    n_cycles = len(signal) // period_int
    return [
        signal[i * period_int:(i + 1) * period_int].astype(np.float64)
        for i in range(n_cycles)
    ]


def extract_cycles_aligned(signal: np.ndarray, period: float,
                           min_period_ratio: float = 0.5,
                           max_period_ratio: float = 1.5) -> dict:
    """Extract cycles aligned to detected peaks or troughs.

    Uses scipy peak detection to find actual cycle boundaries, then extracts
    segments between consecutive extrema. Handles period drift naturally.
    Falls back to rigid extraction if peak detection fails.

    Parameters
    ----------
    signal : 1D time series
    period : estimated period in samples
    min_period_ratio : minimum accepted cycle length as fraction of period
    max_period_ratio : maximum accepted cycle length as fraction of period

    Returns
    -------
    dict with:
        'cycles' : list of 1D arrays (variable length)
        'lengths' : list of cycle lengths
        'n_cycles' : number of extracted cycles
        'method' : 'aligned' or 'rigid' (fallback)
    """
    period_int = max(2, int(np.round(period)))
    signal_f = signal.astype(np.float64)

    if len(signal_f) < 2 * period_int:
        cycles = extract_cycles(signal_f, period)
        return {
            "cycles": cycles,
            "lengths": [period_int] * len(cycles),
            "n_cycles": len(cycles),
            "method": "rigid",
        }

    # Detect peaks and troughs. Prominence threshold is set high enough
    # to avoid finding spurious peaks in noise.
    prominence = max(float(np.std(signal_f)) * 0.3, 1e-6)
    distance = max(1, period_int // 2)

    peaks, _ = sig.find_peaks(signal_f, distance=distance, prominence=prominence)
    troughs, _ = sig.find_peaks(-signal_f, distance=distance, prominence=prominence)

    # Use whichever has more detections
    extrema = peaks if len(peaks) >= len(troughs) else troughs

    if len(extrema) < 3:
        # Not enough extrema — fall back to rigid
        cycles = extract_cycles(signal_f, period)
        return {
            "cycles": cycles,
            "lengths": [period_int] * len(cycles),
            "n_cycles": len(cycles),
            "method": "rigid",
        }

    # Extract cycles between consecutive extrema, filtering by period
    lower = max(2, int(min_period_ratio * period_int))
    upper = max(lower + 1, int(max_period_ratio * period_int))

    cycles = []
    lengths = []
    for i in range(len(extrema) - 1):
        start = int(extrema[i])
        end = int(extrema[i + 1])
        length = end - start
        if length < lower or length > upper:
            continue
        cycles.append(signal_f[start:end])
        lengths.append(length)

    if len(cycles) < 2:
        # Aligned extraction yielded too few — fall back to rigid
        rigid_cycles = extract_cycles(signal_f, period)
        return {
            "cycles": rigid_cycles,
            "lengths": [period_int] * len(rigid_cycles),
            "n_cycles": len(rigid_cycles),
            "method": "rigid",
        }

    return {
        "cycles": cycles,
        "lengths": lengths,
        "n_cycles": len(cycles),
        "method": "aligned",
    }


# ---------------------------------------------------------------------------
# Cycle consistency — now uses aligned extraction with resampling
# ---------------------------------------------------------------------------

def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation with safety checks for flat/short signals."""
    if len(x) < 2 or len(y) < 2:
        return 0.0
    sx, sy = float(np.std(x)), float(np.std(y))
    if sx < 1e-10 or sy < 1e-10:
        return 0.0
    corr = float(np.corrcoef(x, y)[0, 1])
    return 0.0 if np.isnan(corr) else corr


def compute_cycle_consistency(signal: np.ndarray, period: float) -> dict:
    """Compute waveform consistency between successive cycles.

    Uses peak/trough-aligned cycle extraction when possible, falling back
    to rigid equal-length bins. Variable-length cycles are resampled to a
    common length before computing correlations.

    High mean correlation = the waveform shape genuinely repeats.
    Low mean correlation = "things going up and down" with no consistent shape.

    Parameters
    ----------
    signal : 1D time series (measurement window)
    period : estimated period in samples

    Returns
    -------
    dict with:
        'mean_corr' : mean pairwise cycle correlation
        'std_corr' : std of pairwise correlations
        'median_corr' : median pairwise correlation
        'n_cycles' : number of complete cycles found
        'cycle_correlations' : list of all pairwise correlations
        'extraction_method' : 'aligned' or 'rigid'
    """
    _empty = {
        "mean_corr": 0.0,
        "std_corr": 0.0,
        "median_corr": 0.0,
        "n_cycles": 0,
        "cycle_correlations": [],
        "extraction_method": "none",
    }

    period_int = int(np.round(period))
    if period_int < 2 or len(signal) < 2 * period_int:
        return _empty

    # Extract cycles (aligned preferred, rigid fallback)
    extracted = extract_cycles_aligned(signal, period)
    cycles = extracted["cycles"]
    method = extracted["method"]

    if len(cycles) < 2:
        return _empty

    # Resample variable-length cycles to common length for correlation
    if method == "aligned":
        # Use median cycle length as the resample target
        n_resample = max(int(np.median(extracted["lengths"])), 4)
        resampled = []
        for c in cycles:
            if len(c) < 3 or float(np.std(c)) < 1e-10:
                continue
            resampled.append(sig.resample(c, n_resample).astype(np.float64))
        if len(resampled) < 2:
            return _empty
        cycles_for_corr = resampled
    else:
        # Rigid cycles are already equal length
        cycles_for_corr = [c for c in cycles
                           if len(c) >= 2 and float(np.std(c)) > 1e-10]
        if len(cycles_for_corr) < 2:
            return _empty

    # Pairwise Pearson correlation between successive cycles
    correlations = []
    for i in range(len(cycles_for_corr) - 1):
        corr = _safe_corr(cycles_for_corr[i], cycles_for_corr[i + 1])
        correlations.append(corr)

    if not correlations:
        return _empty

    return {
        "mean_corr": float(np.mean(correlations)),
        "std_corr": float(np.std(correlations)),
        "median_corr": float(np.median(correlations)),
        "n_cycles": len(cycles_for_corr),
        "cycle_correlations": correlations,
        "extraction_method": method,
    }


# ---------------------------------------------------------------------------
# Period estimation fallback (for when spectral/ACF gates fail)
# ---------------------------------------------------------------------------

def _estimate_period_from_acf(signal: np.ndarray, fs: float = 1.0,
                              min_lag: int = 3,
                              min_peak_height: float = 0.1) -> tuple[float, float]:
    """Try to estimate period directly from ACF with a low threshold.

    Uses a lower threshold than the main classifier (0.1 vs 0.3) to catch
    slow/noisy rhythms that the main ACF gate rejects.

    Returns (period, acf_peak_height). Returns (0, height) if no peak found.
    """
    x = signal.astype(np.float64)
    x = x - np.mean(x)
    n = len(x)
    if n < 20:
        return 0.0, 0.0

    # FFT-based autocorrelation
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    fft_x = np.fft.rfft(x, n=fft_size)
    acf_full = np.fft.irfft(fft_x * np.conj(fft_x))[:n]
    if acf_full[0] > 0:
        acf_full = acf_full / acf_full[0]
    else:
        return 0.0, 0.0

    # Find peaks in ACF
    max_lag = n // 2
    acf_search = acf_full[min_lag:max_lag]
    if len(acf_search) < 3:
        return 0.0, 0.0

    best_lag, best_height = 0, 0.0
    for i in range(1, len(acf_search) - 1):
        if acf_search[i] > acf_search[i - 1] and acf_search[i] > acf_search[i + 1]:
            if acf_search[i] > best_height:
                best_height = float(acf_search[i])
                best_lag = i + min_lag

    if best_height < min_peak_height or best_lag == 0:
        return 0.0, best_height

    return float(best_lag), best_height


# ---------------------------------------------------------------------------
# Classification — spectral gate removed, period fallback added
# ---------------------------------------------------------------------------

def compute_rhythmicity_index(
    lagged_coherence: float,
    mean_curve_coherence: float,
    mean_cycle_corr: float,
    lagged_coherence_threshold: float = 0.5,
    weak_coherence_threshold: float = 0.3,
    cycle_consistency_threshold: float = 0.4,
    weak_consistency_threshold: float = 0.2,
) -> float:
    """Convert the full class rule into one score aligned with the labels.

    Score bands:
      - < 1.0: ARRHYTHMIC range
      - 1.0 to < 2.0: WEAKLY_RHYTHMIC range
      - >= 2.0: RHYTHMIC range
    """
    eps = 1e-12
    lc = max(float(lagged_coherence), 0.0)
    curve_mean = max(float(mean_curve_coherence), 0.0)
    mean_corr = max(float(mean_cycle_corr), 0.0)

    curve_promote_threshold = max(0.35, weak_coherence_threshold + 0.05)
    weak_curve_threshold = max(0.20, weak_coherence_threshold - 0.05)

    strong_phase_score = max(
        lc / max(float(lagged_coherence_threshold), eps),
        curve_mean / max(float(curve_promote_threshold), eps),
    )
    weak_phase_score = max(
        lc / max(float(weak_coherence_threshold), eps),
        curve_mean / max(float(weak_curve_threshold), eps),
    )
    minimal_phase_score = max(lc, curve_mean) / 0.1

    strong_consistency_score = mean_corr / max(float(cycle_consistency_threshold), eps)
    weak_consistency_score = mean_corr / max(float(weak_consistency_threshold), eps)
    minimal_consistency_score = mean_corr / 0.1

    strong_gate_score = min(strong_phase_score, strong_consistency_score)
    weak_gate_score = max(
        min(weak_phase_score, minimal_consistency_score),
        min(weak_consistency_score, minimal_phase_score),
    )

    if strong_gate_score >= 1.0:
        return float(min(2.0 + (strong_gate_score - 1.0), 2.999))
    if weak_gate_score >= 1.0:
        return float(min(1.0 + (weak_gate_score - 1.0), 1.999))
    return float(max(0.0, min(max(strong_gate_score, weak_gate_score), 0.999)))


def classify_rhythmicity(signal: np.ndarray, fs: float,
                         peak_freq: float, period: float,
                         acf_peak_height: float = 0.0,
                         Q_factor: float = 0.0,
                         has_spectral_peak: bool = True,
                         lagged_coherence_threshold: float = 0.5,
                         weak_coherence_threshold: float = 0.3,
                         cycle_consistency_threshold: float = 0.4,
                         weak_consistency_threshold: float = 0.2) -> dict:
    """Classify rhythmicity by combining lagged coherence and cycle consistency.

    Repetition metrics are always computed when a period estimate is available,
    regardless of whether a spectral peak was detected. If no period is provided,
    attempts ACF-based estimation as a fallback.

    Parameters
    ----------
    signal : 1D time series (measurement window, post-burn-in)
    fs : sampling frequency
    peak_freq : dominant spectral frequency (0 if unknown)
    period : estimated period in samples (0 if unknown)
    acf_peak_height : ACF peak height (existing metric, for reference)
    Q_factor : spectral Q-factor (existing metric, for reference)
    has_spectral_peak : whether a spectral peak was detected above 1/f
    lagged_coherence_threshold : threshold for RHYTHMIC (default 0.5)
    weak_coherence_threshold : threshold for WEAKLY_RHYTHMIC (default 0.3)
    cycle_consistency_threshold : threshold for RHYTHMIC (default 0.4)
    weak_consistency_threshold : threshold for WEAKLY_RHYTHMIC (default 0.2)

    Returns
    -------
    dict with:
        'rhythmicity_class' : RHYTHMIC | WEAKLY_RHYTHMIC | ARRHYTHMIC
        'lagged_coherence' : float — phase consistency across windows
        'lagged_coherence_curve' : dict — coherence vs n_cycles
        'cycle_consistency' : dict — waveform correlation metrics
        'mean_cycle_corr' : float — mean cycle-to-cycle correlation
        'rhythmicity_index' : float — classifier-aligned rhythmicity score
        'Q_factor' : float — (pass-through)
        'acf_peak_height' : float — (pass-through)
        'has_spectral_peak' : bool — (pass-through)
        'period_source' : str — where the period estimate came from
    """
    _empty_curve = {"n_cycles": [], "coherence": [], "mean_coherence": 0.0}
    _empty_cc = {
        "mean_corr": 0.0, "std_corr": 0.0, "median_corr": 0.0,
        "n_cycles": 0, "cycle_correlations": [], "extraction_method": "none",
    }

    def _make_result(rhythmicity_class, lc=0.0, lc_curve=None, cc=None,
                     mean_corr=0.0, rhythmicity_index=0.0, period_source="none"):
        return {
            "rhythmicity_class": rhythmicity_class,
            "lagged_coherence": lc,
            "lagged_coherence_curve": lc_curve or _empty_curve,
            "cycle_consistency": cc or _empty_cc,
            "mean_cycle_corr": mean_corr,
            "rhythmicity_index": rhythmicity_index,
            "Q_factor": Q_factor,
            "acf_peak_height": acf_peak_height,
            "has_spectral_peak": has_spectral_peak,
            "period_source": period_source,
        }

    # --- Resolve period and frequency estimates ---
    # Try to get usable period/freq even without a spectral peak.
    effective_period = period
    effective_freq = peak_freq
    period_source = "none"

    if effective_period > 0 and effective_freq > 0:
        period_source = "provided"
    elif effective_period > 0 and effective_freq <= 0:
        effective_freq = fs / effective_period
        period_source = "provided_period_only"
    elif effective_freq > 0 and effective_period <= 0:
        effective_period = fs / effective_freq
        period_source = "provided_freq_only"
    else:
        # Neither provided — try ACF fallback with low threshold
        fallback_period, fallback_height = _estimate_period_from_acf(
            signal, fs, min_lag=3, min_peak_height=0.1
        )
        if fallback_period > 0:
            effective_period = fallback_period
            effective_freq = fs / fallback_period
            period_source = "acf_fallback"

    # If we still have no period estimate, we genuinely cannot compute
    # repetition metrics — return ARRHYTHMIC with reason.
    if effective_period <= 0 or effective_freq <= 0:
        return _make_result("ARRHYTHMIC", period_source="no_period_estimate")

    # --- Compute repetition metrics ---

    # Lagged coherence at one anchor lag plus the full curve.
    # Using only n_cycles=3 tends to favor very short-period regimes because
    # they generate many more windows than slower rhythms.
    lc = compute_lagged_coherence(signal, fs, effective_freq, n_cycles=3)
    lc_curve = compute_lagged_coherence_curve(signal, fs, effective_freq,
                                               max_n_cycles=10)
    curve_mean_coherence = float(lc_curve["mean_coherence"])

    # Cycle consistency (now using aligned extraction)
    cc = compute_cycle_consistency(signal, effective_period)
    mean_corr = cc["mean_corr"]

    # --- Classification logic ---
    # RHYTHMIC: waveform consistency is strong and coherence evidence is
    # strong either at the anchor lag or across the coherence curve.
    # WEAKLY_RHYTHMIC: at least one metric above weak threshold AND the other
    #   shows some minimal evidence (> 0.1). Prevents classifying pure noise
    #   that accidentally passes one threshold as weakly rhythmic.
    curve_promote_threshold = max(0.35, weak_coherence_threshold + 0.05)
    weak_curve_threshold = max(0.20, weak_coherence_threshold - 0.05)

    strong_phase = (
        lc >= lagged_coherence_threshold
        or curve_mean_coherence >= curve_promote_threshold
    )
    weak_phase = (
        lc >= weak_coherence_threshold
        or curve_mean_coherence >= weak_curve_threshold
    )
    minimal_phase = max(lc, curve_mean_coherence) > 0.1
    rhythmicity_index = compute_rhythmicity_index(
        lagged_coherence=lc,
        mean_curve_coherence=curve_mean_coherence,
        mean_cycle_corr=mean_corr,
        lagged_coherence_threshold=lagged_coherence_threshold,
        weak_coherence_threshold=weak_coherence_threshold,
        cycle_consistency_threshold=cycle_consistency_threshold,
        weak_consistency_threshold=weak_consistency_threshold,
    )

    if strong_phase and mean_corr >= cycle_consistency_threshold:
        rhythmicity_class = "RHYTHMIC"
    elif (weak_phase and mean_corr > 0.1) or \
         (mean_corr >= weak_consistency_threshold and minimal_phase):
        rhythmicity_class = "WEAKLY_RHYTHMIC"
    else:
        rhythmicity_class = "ARRHYTHMIC"

    return _make_result(
        rhythmicity_class,
        lc,
        lc_curve,
        cc,
        mean_corr,
        rhythmicity_index,
        period_source,
    )
