"""Fig 2 - Q-factor vs. true periodicity (3-panel contrast).

Renders three activity traces on a common x-window so they invite direct
comparison: (A) the highest-Q run in the sweep, which is a fast period-2 flip
and NOT a genuine rhythm; (B) a genuine long-period rhythm; (C) a genuine
short-period rhythm. No in-plot annotation - all parameters go in the caption.

Reads the 3 saved traces in ../data/example_traces/. Regenerates without the
simulator. Canonical RI values (from rhythmicity.py) are printed for the caption.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = Path(__file__).resolve().parent.parent / "data" / "example_traces"
OUT = Path(__file__).resolve().parent / "fig2_contrast.png"

# (file stem, common-window slice) - panels share a 1000-step window, ticks every 250
WINDOW = 1000
TICK = 250
PANELS = [
    ("fig2A_maxQ",       "A", 500),   # start index of the 1000-step window
    ("fig2B_longperiod", "B", 0),
    ("fig2C_shortperiod","C", 500),
]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.linewidth": 0.8,
    "xtick.direction": "out", "ytick.direction": "out",
})

fig, axes = plt.subplots(3, 1, figsize=(5.2, 5.4), sharex=True)
for ax, (stem, letter, start) in zip(axes, PANELS):
    trace = np.load(DATA / f"{stem}.npy")
    start = min(start, max(len(trace) - WINDOW, 0))
    seg = trace[start:start + WINDOW]
    t = np.arange(len(seg))
    ax.plot(t, seg, color="#1a1a1a", linewidth=0.7)
    ax.set_ylabel("Firing neurons", fontsize=8)
    ax.set_xlim(0, WINDOW)
    ax.set_xticks(np.arange(0, WINDOW + 1, TICK))
    ax.tick_params(labelsize=7, direction="out")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    # panel letter, just above the top-left corner (clear of ticks)
    ax.text(-0.14, 1.0, letter, transform=ax.transAxes, fontsize=11,
            fontweight="bold", ha="left", va="bottom")
axes[-1].set_xlabel("Timestep", fontsize=8)
fig.tight_layout(h_pad=1.1)
fig.savefig(OUT, dpi=300, bbox_inches="tight")
print(f"saved {OUT}")
