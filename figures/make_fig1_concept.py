"""Figure 1 — the Q-factor is blind to waveform shape; the rhythmicity index is not.

Three *schematic* traces (not simulation output) that share identical regular timing —
the same dominant period of 120 samples in every panel — so their spectral peaks, and
therefore their Q-factors, are essentially the same. What differs across panels is only
shape consistency: whether each cycle's waveform repeats the last.

Construction: a common fundamental sinusoid plus four harmonics whose amplitudes and
phases are re-drawn per cycle. A `mix` parameter interpolates between fully re-randomised
harmonics (panel a — a different waveform every cycle) and a fixed harmonic set (panel c —
one waveform repeating exactly). Panel b sits in between.

The annotated metric values are the illustrative targets for the schematic, matching the
published caption; the point of the figure is the qualitative ordering (Q flat, RI rising),
not a measurement. All simulation figures (2-4) use measured values throughout.

Writes figures/fig1_concept.png
"""
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / "fig1_concept.png"

N = 2400
IDX = np.arange(N)
PERIOD = 120
W = 2 * np.pi / PERIOD
HARMONICS = (2, 3, 4, 5)
FOCAL = "#1f6feb"


def apply_style():
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.size": 8,
        "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 6, "ytick.labelsize": 6,
        "axes.linewidth": 0.6,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.grid": False, "legend.frameon": False,
        "figure.dpi": 200, "savefig.dpi": 300, "savefig.bbox": "tight",
        "axes.titlelocation": "left", "lines.linewidth": 1.2,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def build_trace(seed: int, mix: float, fundamental: float = 2.8,
                harmonic_amp: float = 1.2) -> np.ndarray:
    """Fundamental sinusoid + per-cycle harmonics interpolated toward a fixed set.

    `mix` = 0.0 re-randomises every cycle's harmonics (shape never repeats);
    `mix` = 1.0 uses one fixed harmonic set for every cycle (shape repeats exactly).
    """
    rng = np.random.default_rng(seed)
    rng_fixed = np.random.default_rng(1000 + seed)
    signal = fundamental * np.sin(W * IDX).astype(float)
    fixed = {h: (rng_fixed.uniform(-0.7, 0.7), rng_fixed.uniform(0, 2 * np.pi))
             for h in HARMONICS}
    for cycle in range(N // PERIOD + 1):
        mask = (IDX >= cycle * PERIOD) & (IDX < (cycle + 1) * PERIOD)
        for h in HARMONICS:
            fixed_amp, fixed_phase = fixed[h]
            rand_amp = rng.uniform(-harmonic_amp, harmonic_amp)
            rand_phase = rng.uniform(0, 2 * np.pi)
            amp = mix * fixed_amp + (1 - mix) * rand_amp
            phase = fixed_phase if mix > 0.5 else rand_phase
            signal[mask] += amp * np.sin(h * W * IDX[mask] + phase)
    return signal - signal.mean()


PANELS = {
    "a": dict(mix=0.00, title="Shape varies each cycle",
              subtitle="same regular timing — a different waveform every cycle",
              coherence=1.00, shape=0.35, Q=8.5, RI=2.00),
    "b": dict(mix=0.55, title="Shape mostly repeats",
              subtitle="same regular timing — waveform largely consistent",
              coherence=1.00, shape=0.76, Q=8.5, RI=2.90),
    "c": dict(mix=1.00, title="Shape repeats exactly",
              subtitle="same regular timing — one waveform repeating",
              coherence=1.00, shape=1.00, Q=8.5, RI=3.50),
}

apply_style()
traces = {k: build_trace(4, v["mix"]) for k, v in PANELS.items()}

show = slice(0, 1080)
tt = IDX[show]
fig, axes = plt.subplots(3, 1, figsize=(7.4, 6.6), sharex=True)

for ax, key in zip(axes, "abc"):
    spec = PANELS[key]
    ax.plot(tt, traces[key][show], color=FOCAL, lw=1.4)
    ax.set_yticks([])
    for side in ("left", "top", "right"):
        ax.spines[side].set_visible(False)
    ax.margins(x=0.01, y=0.18)
    ax.text(0.0, 1.17, spec["title"], transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8, fontweight="bold")
    ax.text(0.0, 1.06, spec["subtitle"], transform=ax.transAxes, ha="left", va="top",
            fontsize=6.4, color="0.4")
    label = (f"phase coherence = {spec['coherence']:.2f}\n"
             f"shape consistency = {spec['shape']:.2f}\n"
             f"$Q$ = {spec['Q']:.1f}       RI = {spec['RI']:.2f}")
    ax.text(1.0, 1.17, label, transform=ax.transAxes, ha="right", va="top",
            fontsize=6.3, family="DejaVu Sans Mono", color="0.15",
            bbox=dict(boxstyle="round,pad=0.35", fc="#f4f6fa", ec="0.7", lw=0.5))
    ax.text(-0.06, 1.02, key, transform=ax.transAxes, fontweight="bold",
            fontsize=9, va="bottom", ha="left")

axes[-1].set_xlabel("Time")
fig.suptitle("Q depends only on timing; the rhythmicity index also requires a repeating waveform",
             fontsize=8.6, y=1.0, x=0.01, ha="left")
fig.tight_layout(rect=[0, 0, 1, 0.965])
fig.savefig(OUT, dpi=300, bbox_inches="tight")
print(f"wrote {OUT}")
