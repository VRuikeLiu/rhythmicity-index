"""Figure 4 (revised) — Berger effect under the broadband locked pipeline.

Reads   results/s4_eeg_per_epoch.csv.gz (produced by src/eeg_validation_s4.py +
        src/eeg_score_all_s4.py)
Writes  figures/fig4_eeg.png

Plots subject-level means of the primary variant (broadband 1-45 Hz, free peak
estimation, artifact rejection, admissible epochs only): paired subjects as
connected filled circles, eyes-closed-only subjects as open circles, condition
means with 95% CI, and the RI=1 / RI=2 category boundaries.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
EP = ROOT / "results" / "s4_eeg_per_epoch.csv.gz"
OUT = Path(__file__).resolve().parent / "fig4_eeg.png"

allep = pd.read_csv(EP)
acc = allep[~allep.rejected & allep.admissible_v1.fillna(False)]
sm = acc.groupby(["subject", "condition"]).RI_v1.mean().reset_index()
w = sm.pivot(index="subject", columns="condition", values="RI_v1")
paired = w.dropna()
unpaired_ec = w[w.EO.isna()].EC

eo, ec = paired.EO.to_numpy(), paired.EC.to_numpy()
diff = ec - eo
tt = stats.ttest_rel(ec, eo)
wc = stats.wilcoxon(ec, eo)
ci = stats.t.interval(0.95, len(diff) - 1, loc=diff.mean(), scale=stats.sem(diff))
dz = diff.mean() / diff.std(ddof=1)
print(f"n={len(paired)} paired | EO {eo.mean():.3f} EC {ec.mean():.3f} "
      f"delta {diff.mean():.3f} CI [{ci[0]:.3f},{ci[1]:.3f}] "
      f"t={tt.statistic:.2f} p={tt.pvalue:.2g} W={wc.statistic:.0f} p={wc.pvalue:.2g} "
      f"dz={dz:.2f} up={int((diff > 0).sum())}/{len(diff)}")

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                     "axes.linewidth": 0.8, "axes.spines.top": False,
                     "axes.spines.right": False})

def ci95(v):
    v = np.asarray(v)
    return v.mean(), stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))

rng = np.random.default_rng(4)
x_eo, x_ec, off = 0.0, 1.0, 0.22
jp = rng.uniform(-0.07, 0.07, len(paired))
ju = rng.uniform(-0.07, 0.07, len(unpaired_ec))

fig, ax = plt.subplots(figsize=(4.8, 3.9))
for (idx, row), j in zip(paired.iterrows(), jp):
    ax.plot([x_eo + j, x_ec + j], [row.EO, row.EC], "-", color="0.75", lw=0.8, zorder=1)
ax.plot(x_eo + jp, paired.EO, "o", color="0.25", ms=4.5, zorder=3)
ax.plot(x_ec + jp, paired.EC, "o", color="0.25", ms=4.5, zorder=3)
ax.plot(x_ec + ju, unpaired_ec, "o", mfc="white", mec="0.45", ms=4.5, zorder=2)
m_eo, h_eo = ci95(paired.EO)
m_ec, h_ec = ci95(paired.EC)
ax.errorbar([x_eo - off, x_ec + off], [m_eo, m_ec], yerr=[[h_eo, h_ec], [h_eo, h_ec]],
            fmt="s-", color="#b03a2e", lw=2.0, ms=6, capsize=3.5, zorder=4,
            label="condition mean ± 95% CI (n = 15 paired)")
for yv, lab in ((1.0, "RI = 1 (weakly rhythmic)"), (2.0, "RI = 2 (rhythmic)")):
    ax.axhline(yv, color="0.55", lw=0.8, ls=(0, (4, 3)), zorder=0)
    ax.annotate(lab, xy=(1.62, yv), xytext=(0, 3), textcoords="offset points",
                va="bottom", ha="left", fontsize=6, color="0.35")
ax.set_xticks([x_eo, x_ec])
ax.set_xticklabels(["Eyes open", "Eyes closed"])
ax.set_xlim(-0.42, 2.28)
ax.set_ylim(-0.06, 3.05)
ax.set_ylabel("Mean rhythmicity index per subject", labelpad=8)
ax.plot([], [], "o", mfc="white", mec="0.45", ms=4.5,
        label="eyes-closed only (no artifact-free\neyes-open epochs; excluded from tests)")
ax.legend(loc="upper left", frameon=False, handlelength=1.6, fontsize=6.5)
fig.tight_layout()
fig.savefig(OUT, dpi=300, bbox_inches="tight")
print(f"wrote {OUT}")
