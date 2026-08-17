"""Ground-truth benchmark battery for the rhythmicity index (reviewer item MF-2b).

The reviewer asked for an external benchmark with known ground truth: sustained and
damped oscillations, bursts, chirps, phase-jittered and amplitude-modulated rhythms,
1/f noise, AR(2) narrowband noise, and spectrum-matched surrogates, with sensitivity,
specificity and ROC reported on held-out signals.

HELD-OUT DISCIPLINE
-------------------
Nothing in the index was fit to these signals: the estimator constants were locked in
Session 1 (rhythmicity_locked.py) before this battery existed, and every family's
parameter grid below was fixed at design time. Generator correctness was checked on
seeds < 1000; every reported number comes from evaluation seeds >= 10_000 that were
never inspected during design. The RI thresholds used at the operating points are the
manuscript's fixed constants (RI >= 2 rhythmic, RI >= 1 weakly rhythmic).

GROUND-TRUTH LABELS (by construction)
-------------------------------------
positive = the signal contains a waveform that genuinely repeats (possibly degraded
by noise, amplitude modulation, timing jitter, bursting, damping, or frequency drift).
negative = a stochastic process with no temporally consistent repeating waveform.

Two label subsets are reported:
  core     : stationary rhythms (sustained, AM, timing jitter <= 2 %) vs all negatives.
             These are unambiguously inside the index's declared scope (a fixed-period,
             sustained, shape-repeating rhythm).
  all      : additionally counts non-stationary rhythms (heavy jitter, bursts, damped,
             chirps) as positives. The index measures whole-window sustained
             rhythmicity, so misses here bound its scope rather than reveal errors.

A NOTE ON SPECTRUM-MATCHED SURROGATES. A phase-randomised (Theiler) surrogate of a
STATIONARY periodic signal is itself near-periodic: the harmonic magnitudes survive,
only their relative phases change, so the surrogate has a repeating (different) shape
and is genuinely rhythmic. Surrogates are therefore built from the jittered and bursty
arms, whose broadened spectra randomise to stationary narrowband noise with no
repeating waveform (label: negative). Surrogates of the sustained arm are kept as a
separate DIAGNOSTIC arm excluded from the ROC, with the expectation that a correct
shape-sensitive index keeps scoring them rhythmic.

All signals: n = 1500 samples at fs = 1 (matching the simulation measurement window),
fundamental period drawn log-uniform in [16, 100] samples so every instance clears the
admissibility floors (>= 8 samples/cycle, >= 10 cycles) by construction.
"""
from __future__ import annotations

import numpy as np

N_SAMPLES = 1500
P_RANGE = (16.0, 100.0)          # log-uniform fundamental period, samples
N_HARM = 3                       # band-limited nonsinusoidal waveform
SEED_EXPLORE = 0                 # design-time sanity checks only
SEED_HELDOUT = 10_000            # evaluation block, never inspected during design

# --- waveform primitives -------------------------------------------------------


def _draw_shape(rng):
    """Random band-limited waveform: 3 harmonics, unit-amplitude fundamental."""
    amps = np.r_[1.0, rng.uniform(0.2, 0.7, N_HARM - 1)]
    phs = rng.uniform(0, 2 * np.pi, N_HARM)
    return amps, phs


def _eval_shape(phase, amps, phs):
    x = np.zeros_like(phase)
    for h in range(N_HARM):
        x += amps[h] * np.sin((h + 1) * phase + phs[h])
    return x


def _draw_period(rng):
    lo, hi = np.log(P_RANGE[0]), np.log(P_RANGE[1])
    return float(np.exp(rng.uniform(lo, hi)))


def _norm(x):
    s = x.std()
    return x / s if s > 0 else x


# --- positive families ----------------------------------------------------------


def gen_sustained(rng, noise_sigma):
    P = _draw_period(rng)
    amps, phs = _draw_shape(rng)
    t = np.arange(N_SAMPLES)
    x = _norm(_eval_shape(2 * np.pi * t / P, amps, phs))
    return x + noise_sigma * rng.standard_normal(N_SAMPLES), P


def gen_am(rng, depth, noise_sigma=0.2):
    P = _draw_period(rng)
    amps, phs = _draw_shape(rng)
    t = np.arange(N_SAMPLES)
    T_env = P * rng.uniform(8, 15)
    env = 1.0 + depth * np.sin(2 * np.pi * t / T_env + rng.uniform(0, 2 * np.pi))
    x = _norm(_eval_shape(2 * np.pi * t / P, amps, phs) * env)
    return x + noise_sigma * rng.standard_normal(N_SAMPLES), P


def _jitter_phase(rng, P, jitter):
    """Piecewise-linear phase with cycle lengths ~ N(P, (jitter*P)^2)."""
    lengths = []
    total = 0.0
    while total < N_SAMPLES + 2 * P:
        L = max(0.5 * P, rng.normal(P, jitter * P))
        lengths.append(L)
        total += L
    bounds = np.r_[0.0, np.cumsum(lengths)]
    t = np.arange(N_SAMPLES, dtype=float)
    idx = np.searchsorted(bounds, t, side="right") - 1
    frac = (t - bounds[idx]) / np.asarray(lengths)[idx]
    return 2 * np.pi * (idx + frac)


def gen_jitter(rng, jitter, noise_sigma=0.2):
    P = _draw_period(rng)
    amps, phs = _draw_shape(rng)
    phase = _jitter_phase(rng, P, jitter)
    x = _norm(_eval_shape(phase, amps, phs))
    return x + noise_sigma * rng.standard_normal(N_SAMPLES), P


def gen_burst(rng, duty, noise_sigma=0.3):
    """Whole cycles gated on/off in runs; background noise continues in gaps."""
    P = _draw_period(rng)
    amps, phs = _draw_shape(rng)
    n_cyc = int(np.ceil(N_SAMPLES / P)) + 1
    on = np.zeros(n_cyc, bool)
    i = 0
    while i < n_cyc:
        run = max(1, int(round(rng.uniform(5, 10))))
        if rng.random() < duty:
            on[i:i + run] = True
        i += run
    if not on.any():
        on[: max(1, n_cyc // 3)] = True
    t = np.arange(N_SAMPLES)
    cyc_idx = np.floor(t / P).astype(int).clip(0, n_cyc - 1)
    x = _eval_shape(2 * np.pi * t / P, amps, phs) * on[cyc_idx]
    return _norm(x) + noise_sigma * rng.standard_normal(N_SAMPLES), P


def gen_damped(rng, tau_cycles, noise_sigma=0.2):
    P = _draw_period(rng)
    amps, phs = _draw_shape(rng)
    t = np.arange(N_SAMPLES)
    onset = int(rng.uniform(0, 2 * P))
    env = np.zeros(N_SAMPLES)
    env[onset:] = np.exp(-(t[onset:] - onset) / (tau_cycles * P))
    x = _eval_shape(2 * np.pi * t / P, amps, phs) * env
    return _norm(x) + noise_sigma * rng.standard_normal(N_SAMPLES), P


def gen_chirp(rng, sweep, noise_sigma=0.2):
    """Linear frequency sweep f0*(1-s/2) -> f0*(1+s/2), harmonic waveform."""
    P = _draw_period(rng)
    amps, phs = _draw_shape(rng)
    f0 = 1.0 / P
    t = np.arange(N_SAMPLES, dtype=float)
    f_start = f0 * (1 - sweep / 2)
    rate = f0 * sweep / N_SAMPLES
    phase = 2 * np.pi * (f_start * t + 0.5 * rate * t ** 2)
    x = _norm(_eval_shape(phase, amps, phs))
    return x + noise_sigma * rng.standard_normal(N_SAMPLES), P


def gen_shape_randomized(rng, noise_sigma=0.2):
    """Fixed cycle timing, fresh band-limited waveform every cycle.

    'Things going up and down' with a regular clock but no consistent shape --
    the direct control for the index's shape term.
    """
    P = _draw_period(rng)
    t = np.arange(N_SAMPLES)
    cyc_idx = np.floor(t / P).astype(int)
    phase = 2 * np.pi * (t / P - cyc_idx)
    x = np.zeros(N_SAMPLES)
    for ci in range(cyc_idx.max() + 1):
        m = cyc_idx == ci
        amps, phs = _draw_shape(rng)
        seg = _eval_shape(phase[m], amps, phs)
        x[m] = seg - seg.mean()
    return _norm(x) + noise_sigma * rng.standard_normal(N_SAMPLES), P


# --- negative families ----------------------------------------------------------


def gen_powerlaw(rng, gamma):
    wh = rng.standard_normal(N_SAMPLES)
    F = np.fft.rfft(wh)
    f = np.fft.rfftfreq(N_SAMPLES, 1.0)
    f[0] = f[1]
    F *= f ** (-gamma / 2.0)
    x = np.fft.irfft(F, N_SAMPLES)
    return _norm(x), np.nan


def gen_ar2(rng, rho, warmup=500):
    f0 = 1.0 / _draw_period(rng)
    w0 = 2 * np.pi * f0
    a1, a2 = 2 * rho * np.cos(w0), -rho ** 2
    e = rng.standard_normal(N_SAMPLES + warmup)
    x = np.zeros(N_SAMPLES + warmup)
    for i in range(2, N_SAMPLES + warmup):
        x[i] = a1 * x[i - 1] + a2 * x[i - 2] + e[i]
    return _norm(x[warmup:]), 1.0 / f0


def phase_randomize(x, rng):
    F = np.fft.rfft(x)
    mag = np.abs(F)
    ph = rng.uniform(0, 2 * np.pi, len(F))
    ph[0] = np.angle(F[0])
    if N_SAMPLES % 2 == 0:
        ph[-1] = np.angle(F[-1])
    return _norm(np.fft.irfft(mag * np.exp(1j * ph), N_SAMPLES))


# --- battery definition ----------------------------------------------------------
# (family, arm parameter, n instances, label, in stationary core, generator)

def battery_spec():
    spec = []
    for s in (0.1, 0.2, 0.3, 0.5):
        spec.append(("sustained", f"sigma={s}", 50, "pos", True,
                     lambda r, s=s: gen_sustained(r, s)))
    for m in (0.25, 0.5, 0.75, 1.0):
        spec.append(("am", f"depth={m}", 50, "pos", True,
                     lambda r, m=m: gen_am(r, m)))
    for j in (0.01, 0.02, 0.05, 0.10):
        core = j <= 0.02
        spec.append(("jitter", f"j={j}", 50, "pos", core,
                     lambda r, j=j: gen_jitter(r, j)))
    for d in (0.3, 0.5, 0.7):
        spec.append(("burst", f"duty={d}", 66, "pos", False,
                     lambda r, d=d: gen_burst(r, d)))
    for tau in (5, 10, 20):
        spec.append(("damped", f"tau={tau}", 66, "pos", False,
                     lambda r, tau=tau: gen_damped(r, tau)))
    for sw in (0.05, 0.1, 0.2, 0.4):
        spec.append(("chirp", f"sweep={sw}", 50, "pos", False,
                     lambda r, sw=sw: gen_chirp(r, sw)))
    spec.append(("white", "gamma=0", 100, "neg", True,
                 lambda r: gen_powerlaw(r, 0.0)))
    spec.append(("pink", "gamma=1", 100, "neg", True,
                 lambda r: gen_powerlaw(r, 1.0)))
    spec.append(("brown", "gamma=2", 100, "neg", True,
                 lambda r: gen_powerlaw(r, 2.0)))
    for rho in (0.90, 0.95, 0.98, 0.99):
        spec.append(("ar2", f"rho={rho}", 50, "neg", True,
                     lambda r, rho=rho: gen_ar2(r, rho)))
    # spectrum-matched surrogates of broadband-ised rhythms (see module docstring)
    for j in (0.05, 0.10):
        spec.append(("surrogate_jitter", f"j={j}", 50, "neg", True,
                     lambda r, j=j: (phase_randomize(gen_jitter(r, j)[0], r), np.nan)))
    spec.append(("surrogate_burst", "duty=0.5", 100, "neg", True,
                 lambda r: (phase_randomize(gen_burst(r, 0.5)[0], r), np.nan)))
    spec.append(("shape_randomized", "fresh-cycle", 200, "neg", True,
                 gen_shape_randomized))
    # diagnostic arm, excluded from ROC (see docstring)
    spec.append(("surrogate_sustained", "sigma=0.2", 50, "diag", False,
                 lambda r: (phase_randomize(gen_sustained(r, 0.2)[0], r), np.nan)))
    return spec


def run_battery(analyze, seed_base):
    """Generate every battery instance and score it. Returns list of dicts."""
    rows = []
    k = 0
    for family, arm, n, label, core, gen in battery_spec():
        for i in range(n):
            rng = np.random.default_rng(seed_base + k)
            x, P_true = gen(rng)
            res = analyze(x)
            rows.append(dict(family=family, arm=arm, truth=label, core=core,
                             instance=i, seed=seed_base + k, P_true=P_true,
                             ri_label=res["label"],
                             **{kk: res[kk] for kk in
                                ("RI", "admissible", "reason", "Q",
                                 "peak_power_share", "c", "cbar", "r", "period",
                                 "samples_per_cycle", "n_cycles",
                                 "freq_precision_ok")}))
            k += 1
    return rows
