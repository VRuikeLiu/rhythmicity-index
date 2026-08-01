"""Figure 4 — the rhythmicity index transfers to human EEG (Berger effect).

Per-subject mean rhythmicity index for eyes-open versus eyes-closed conditions, from the
EEG Motor Movement/Imagery Database (EEGMMIDB, PhysioNet): 20 subjects, 600 four-second
epochs, occipital channels (O1/Oz/O2), alpha band 8-13 Hz. Grey lines connect the two
conditions within a subject; the red line is the condition mean.

This is the paper's external validation: the index was developed entirely on simulated
signals and applied unchanged to real recordings, where the expected answer is
independently established (eyes-closed occipital alpha is rhythmic, eyes-open is not).

The per-subject values are produced by src/eeg_validation.py (which downloads the EEG via
mne.datasets.eegbci); this script only plots the resulting table and prints the summary
statistics it computes from it.

Reads   data/eeg_rhythmicity.csv
Writes  figures/fig4_eeg.png
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "eeg_rhythmicity.csv"
OUT = Path(__file__).resolve().parent / "fig4_eeg.png"

df = pd.read_csv(DATA)
wide = df.pivot(index="subject", columns="condition", values="mean_RI")

eo, ec = wide["EO"].to_numpy(), wide["EC"].to_numpy()
diff = ec - eo
n_up = int((diff > 0).sum())
d_paired = float(diff.mean() / diff.std(ddof=1))
d_pooled = float((ec.mean() - eo.mean())
                 / np.sqrt((eo.var(ddof=1) + ec.var(ddof=1)) / 2.0))

print(f"n = {len(wide)} subjects, {int(df['n_epochs'].sum())} epochs")
print(f"EO mean RI = {eo.mean():.3f}   EC mean RI = {ec.mean():.3f}   "
      f"delta = {ec.mean() - eo.mean():.3f}")
print(f"{n_up}/{len(wide)} subjects increase when the eyes close")
print(f"paired Cohen's dz = {d_paired:.2f}   pooled Cohen's d = {d_pooled:.2f}")

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.linewidth": 0.8,
                     "xtick.direction": "out", "ytick.direction": "out"})

fig, ax = plt.subplots(figsize=(4.6, 3.8))
x = [0.0, 1.0]
for a, b in zip(eo, ec):
    ax.plot(x, [a, b], "-", color="0.72", lw=0.8, zorder=1)
ax.plot([x[0]] * len(eo), eo, "o", color="0.28", markersize=5, zorder=2)
ax.plot([x[1]] * len(ec), ec, "o", color="0.28", markersize=5, zorder=2)
ax.plot(x, [eo.mean(), ec.mean()], "-", color="#c0392b", lw=2.6, zorder=3)

ax.set_xticks(x)
ax.set_xticklabels(["Eyes open", "Eyes closed"])
ax.set_xlim(-0.28, 1.28)
ax.set_ylabel("Mean rhythmicity index")
for side in ("top", "right"):
    ax.spines[side].set_visible(False)

fig.tight_layout()
fig.savefig(OUT, dpi=300, bbox_inches="tight")
print(f"wrote {OUT}")
