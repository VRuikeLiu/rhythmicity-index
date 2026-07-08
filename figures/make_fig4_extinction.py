"""Fig 4 - Extinction / survival threshold vs alpha (single clean panel).

Survival probability (= 1 - extinction_prob) as a function of alpha on a single
frozen network, showing the sharp boundary near alpha ~ 0.007. Single panel;
the forest-plot / ANOVA / per-network CV panels are intentionally dropped.

Reads ../data/run_summary.csv. Caption reports alpha_crit and the Kinouchi-Copelli
branching-ratio context (see figure_render_spec).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = Path(__file__).resolve().parent.parent / "data" / "run_summary.csv"
OUT = Path(__file__).resolve().parent / "fig4_extinction.png"

df = pd.read_csv(DATA)
# Single frozen network + location, representative beta (0.05), n=1 fixed init.
sub = df[(df["network"] == "net_A") & (df["location_idx"] == 0) & np.isclose(df["beta"], 0.05)]
sub = sub.sort_values("alpha")
alpha = sub["alpha"].to_numpy()
survival = 1.0 - sub["extinction_prob"].to_numpy()

# alpha_crit: first alpha where survival crosses 0.5
cross = np.where(survival >= 0.5)[0]
alpha_crit = float(alpha[cross[0]]) if len(cross) else np.nan

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.linewidth": 0.8,
                     "xtick.direction": "out", "ytick.direction": "out"})
fig, ax = plt.subplots(figsize=(4.6, 3.2))
ax.plot(alpha, survival, "-o", color="#1a1a1a", linewidth=1.1, markersize=3.5)
if np.isfinite(alpha_crit):
    ax.axvline(alpha_crit, color="#c0392b", linewidth=1.0, linestyle="--")
ax.set_xscale("log")
ax.set_xlabel(r"$\alpha$ (ignition probability per active neighbor)", fontsize=8)
ax.set_ylabel("Survival probability", fontsize=8)
ax.set_ylim(-0.03, 1.03)
ax.tick_params(labelsize=7, direction="out")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(OUT, dpi=300, bbox_inches="tight")
print(f"saved {OUT} | alpha_crit (survival>=0.5) = {alpha_crit:.5f}")
