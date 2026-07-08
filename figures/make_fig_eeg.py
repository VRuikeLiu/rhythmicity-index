"""Fig 3 - EEG Berger validation (per-subject eyes-open vs eyes-closed RI).

Paired per-subject comparison of mean rhythmicity index for eyes-closed (EC)
vs eyes-open (EO), on EEGMMIDB (PhysioNet), 20 subjects. Clean styling to match
Fig 2 / Fig 4; no internal labels.

Reads ../data/eeg_rhythmicity.csv.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = Path(__file__).resolve().parent.parent / "data" / "eeg_rhythmicity.csv"
OUT = Path(__file__).resolve().parent / "fig3_eeg.png"

df = pd.read_csv(DATA)
piv = df.pivot(index="subject", columns="condition", values="mean_RI").dropna()

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.linewidth": 0.8,
                     "xtick.direction": "out", "ytick.direction": "out"})
fig, ax = plt.subplots(figsize=(4.2, 3.4))
x = [0, 1]
for _, row in piv.iterrows():
    ax.plot(x, [row["EO"], row["EC"]], "-", color="#9aa0a6", linewidth=0.7, alpha=0.7, zorder=1)
    ax.plot(x, [row["EO"], row["EC"]], "o", color="#4a4a4a", markersize=3, zorder=2)
# condition means
ax.plot(x, [piv["EO"].mean(), piv["EC"].mean()], "-", color="#c0392b", linewidth=2.0, zorder=3)
ax.set_xticks(x)
ax.set_xticklabels(["Eyes open", "Eyes closed"], fontsize=8)
ax.set_xlim(-0.35, 1.35)
ax.set_ylabel("Mean rhythmicity index", fontsize=8)
ax.tick_params(labelsize=7, direction="out")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(OUT, dpi=300, bbox_inches="tight")
d = (piv["EC"].mean() - piv["EO"].mean())
print(f"saved {OUT} | EO mean={piv['EO'].mean():.3f} EC mean={piv['EC'].mean():.3f} delta={d:.3f}")
