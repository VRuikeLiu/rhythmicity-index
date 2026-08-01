"""Figure 1 — the Q-factor is blind to waveform shape; the rhythmicity index is not.

Three *schematic* traces (not simulation output) that share identical regular timing —
the same dominant period of 120 samples in every panel — so their spectral peaks, and
therefore their Q-factors, are essentially the same. What differs across panels is only
shape consistency: whether each cycle's waveform repeats the last.

Construction: a common fundamental sinusoid plus four harmonics whose amplitudes and
phases are re-drawn per cycle. A `mix` parameter interpolates between fully re-randomised
harmonics (panel a — a different waveform every cycle) and a fixed harmonic set (panel c —
one waveform repeating exactly). Panel b sits in between.

Every annotated value is MEASURED here at render time by `src/analyze.py` on the trace that
is plotted — same as Figures 3 and 4, nothing hardcoded. The script prints what it measured.

Spectral resolution for this figure
-----------------------------------
These traces have a fundamental period of 120 samples, so the paper's default Welch setting
(`nperseg=256`, chosen for the simulation traces whose periods are 2-6 timesteps) gives only
two cycles per segment and cannot resolve the peak: it reports Q = 1.0 for all three panels.
This figure therefore measures Q with `nperseg=2048` (~17 cycles per segment), which resolves
the fundamental and yields Q = 8.5 — identical across all three panels, which is precisely the
figure's point. The rhythmicity index is measured at the same setting. See the README note on
Q being resolution-dependent: the setting must suit the period of the signal in hand.

Writes figures/fig1_concept.png
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from analyze import analyze  # noqa: E402

OUT = Path(__file__).resolve().parent / "fig1_concept.png"

#: Welch segment length suited to the 120-sample fundamental of these schematic traces.
NPERSEG = 2048

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
              subtitle="same regular timing — a different waveform every cycle"),
    "b": dict(mix=0.55, title="Shape mostly repeats",
              subtitle="same regular timing — waveform largely consistent"),
    "c": dict(mix=1.00, title="Shape repeats exactly",
              subtitle="same regular timing — one waveform repeating"),
}

apply_style()
traces = {k: build_trace(4, v["mix"]) for k, v in PANELS.items()}

# Measure each plotted trace — Q, phase coherence, shape consistency and RI.
measured = {k: analyze(traces[k], fs=1.0, nperseg=NPERSEG) for k in PANELS}

print(f"measured with nperseg={NPERSEG} (fundamental period {PERIOD} samples):")
for k in "abc":
    m = measured[k]
    print(f"  {k}  Q = {m['Q']:.2f}   phase coherence = {m['lagged_coherence']:.2f}   "
          f"shape consistency = {m['mean_cycle_corr']:.2f}   RI = {m['RI']:.2f} "
          f"({m['rhythmicity_class']})")
q_values = {round(measured[k]["Q"], 2) for k in "abc"}
print(f"  Q is {'identical' if len(q_values) == 1 else 'NOT identical'} across panels: "
      f"{sorted(q_values)}  <- the figure's point")

show = slice(0, 1080)
tt = IDX[show]
fig, axes = plt.subplots(3, 1, figsize=(7.4, 6.6), sharex=True)

for ax, key in zip(axes, "abc"):
    spec = PANELS[key]
    m = measured[key]
    ax.plot(tt, traces[key][show], color=FOCAL, lw=1.4)
    ax.set_yticks([])
    for side in ("left", "top", "right"):
        ax.spines[side].set_visible(False)
    ax.margins(x=0.01, y=0.18)
    ax.text(0.0, 1.17, spec["title"], transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8, fontweight="bold")
    ax.text(0.0, 1.06, spec["subtitle"], transform=ax.transAxes, ha="left", va="top",
            fontsize=6.4, color="0.4")
    label = (f"phase coherence = {m['lagged_coherence']:.2f}\n"
             f"shape consistency = {m['mean_cycle_corr']:.2f}\n"
             f"$Q$ = {m['Q']:.1f}       RI = {m['RI']:.2f}")
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
