"""Figure 2 — the generator shows the expected sharp extinct-active transition.

Survival probability (fraction of runs with self-sustaining activity, = 1 - extinction
probability) versus the per-edge firing probability alpha, on a single frozen network
(net_A, location 0, beta = 0.05, one fixed initial firing node; log-scaled alpha).

The marked critical point is the last fully-extinct alpha in the sweep (alpha_c = 0.0042).
It agrees with the mean-field branching-ratio prediction alpha ~ 1/K = 0.005 for K = 200
(Kinouchi & Copelli 2006). At the right edge alpha = 1 is the degenerate deterministic
limit: every resting node with an active neighbour fires at the same step, the whole
active population enters refractoriness together, and activity self-extinguishes.

Reads   data/run_summary.csv
Writes  figures/fig2_extinction.png
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "run_summary.csv"
OUT = Path(__file__).resolve().parent / "fig2_extinction.png"

NETWORK, LOCATION, BETA = "net_A", 0, 0.05
ALPHA_CRIT = 0.004217  # last fully-extinct alpha in the sweep; printed as 0.0042

df = pd.read_csv(DATA)
sub = df[(df["network"] == NETWORK)
         & (df["location_idx"] == LOCATION)
         & np.isclose(df["beta"], BETA)].sort_values("alpha")

alpha = sub["alpha"].to_numpy()
survival = 1.0 - sub["extinction_prob"].to_numpy()

i_crit = int(np.where(np.isclose(alpha, ALPHA_CRIT, atol=1e-4))[0][0])
a_crit, s_crit = alpha[i_crit], survival[i_crit]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                     "axes.linewidth": 0.8,
                     "xtick.direction": "out", "ytick.direction": "out"})

fig, ax = plt.subplots(figsize=(4.8, 3.2))
ax.plot(alpha, survival, "-o", color="#1a1a1a", linewidth=1.1, markersize=3.5, zorder=2)
ax.plot([a_crit], [s_crit], "o", markersize=9, markerfacecolor="none",
        markeredgecolor="#c0392b", markeredgewidth=1.8, zorder=4)
ax.annotate(r"critical point ($\alpha_\mathrm{c}=0.0042$)",
            xy=(a_crit, s_crit), xytext=(a_crit * 1.05, 0.32),
            fontsize=7.2, color="#c0392b",
            arrowprops=dict(arrowstyle="-", color="#c0392b", lw=0.8))

ax.set_xscale("log")
ax.set_xlabel(r"$\alpha$ (ignition probability per active neighbor)", fontsize=8)
ax.set_ylabel("Survival probability", fontsize=8)
ax.set_ylim(-0.03, 1.03)
ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100))
ax.xaxis.set_minor_formatter(NullFormatter())
ax.tick_params(axis="x", which="major", labelsize=8, length=4)
ax.tick_params(axis="x", which="minor", length=2)
ax.tick_params(axis="y", labelsize=7)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)

fig.tight_layout()
fig.savefig(OUT, dpi=300, bbox_inches="tight")
print(f"wrote {OUT}  |  alpha_c = {a_crit:.6f}, survival at alpha=1 is {survival[-1]:.2f}")
