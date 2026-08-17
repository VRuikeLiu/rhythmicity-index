"""Figure 3 — spectral sharpness does not track genuine repetition.

Three panels:

  A) the joint distribution of Q and RI over the 11,938 admissible surviving runs
     of the 125,000-run sweep (Spearman rho = -0.005) — spectral sharpness carries
     no rank information about waveform repetition;
  B) a sharp-spectrum comparison run (alpha = 0.421697, beta = 0.05; Q = 12.49,
     the 94th percentile of admissible runs) whose waveform does not repeat
     (RI = 1.342, weakly rhythmic);
  C) the highest-RI admissible run in the sweep (alpha = 0.013335, beta = 0.05;
     RI = 2.378 at a measured period of 26.08 timesteps) whose spectrum is
     unremarkable (Q = 2.398).

B and C share every design coordinate except alpha: same network realisation
(net_C, graph seed 14345), same seed location (node 0), same beta, same replicate
index — a true single-parameter contrast.

Both traces ship in data/example_traces/ AND are regenerable in isolation from
their design-position seeds (run_seed 54202 and 51802; see sweep_design.py and
sweep_sim.py) — regeneration is bitwise-identical to the shipped traces. The
scatter reads results/per_run_results.csv.gz. Panel annotations quote the sweep's
delivery-table values (results/), which are the paper's record; the script also
re-measures both traces with the locked estimator and asserts Q, c and the period
agree with the table (the shape term r can differ in the last digits across SciPy
builds — see README "Reproducibility notes").

Reads   results/per_run_results.csv.gz, data/example_traces/fig3{B,C}_*.npy
Writes  figures/fig3_contrast.png
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rhythmicity_locked_freq import analyze_s2   # noqa: E402

OUT = Path(__file__).resolve().parent / "fig3_contrast.png"
PER_RUN = ROOT / "results" / "per_run_results.csv.gz"
TRACE_B = ROOT / "data" / "example_traces" / "fig3B_sharpQ.npy"
TRACE_C = ROOT / "data" / "example_traces" / "fig3C_highestRI.npy"

WIN = (500, 2000)      # stationary measurement window
DET = (1000, 1150)     # 150-step detail
SEED_B, SEED_C = 54202, 51802
COL_B, COL_C, GREY = "#4477AA", "#EE7733", "#BBBBBB"


def apply_style():
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.size": 8,
        "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 6, "ytick.labelsize": 6,
        "axes.linewidth": 0.6,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 3, "ytick.major.size": 3,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 300, "savefig.dpi": 300,
    })


def main():
    apply_style()
    df = pd.read_csv(PER_RUN)
    adm = df[df.admissible == True]                                   # noqa: E712
    rowB = df.loc[df.run_seed == SEED_B].iloc[0]
    rowC = df.loc[df.run_seed == SEED_C].iloc[0]
    trB = np.load(TRACE_B)[WIN[0]:WIN[1]]
    trC = np.load(TRACE_C)[WIN[0]:WIN[1]]

    # re-measure and check against the delivery table (paper record)
    for tr, row, name in [(trB, rowB, "B"), (trC, rowC, "C")]:
        res = analyze_s2(tr.astype(np.float64), fs=1.0)
        for k in ("Q", "c", "period"):
            assert np.isclose(res[k], row[k], rtol=0, atol=1e-9), (name, k, res[k], row[k])
        print(f"panel {name}: table Q={row.Q:.3f} RI={row.RI:.3f} "
              f"(re-measured Q={res['Q']:.3f} RI={res['RI']:.3f})")

    t = np.arange(*WIN)
    fig = plt.figure(figsize=(7.08, 6.3))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.45, 1, 1], width_ratios=[3.2, 1],
                          hspace=0.42, wspace=0.06,
                          left=0.085, right=0.985, top=0.94, bottom=0.075)

    axA = fig.add_subplot(gs[0, :])
    axA.scatter(adm.Q, adm.RI, s=3, c=GREY, alpha=0.25, linewidths=0, rasterized=True)
    axA.set_xscale("log")
    for y, lab in [(2, "RI = 2 (rhythmic threshold)"), (1, "RI = 1 (arrhythmic below)")]:
        axA.axhline(y, ls="--", lw=0.8, c="0.35")
        axA.annotate(lab, xy=(0.99, y + 0.04), xycoords=("axes fraction", "data"),
                     ha="right", va="bottom", fontsize=6, color="0.25")
    axA.scatter([rowB.Q], [rowB.RI], s=42, c=COL_B, edgecolors="k", linewidths=0.6, zorder=5)
    axA.scatter([rowC.Q], [rowC.RI], s=70, c=COL_C, edgecolors="k", linewidths=0.6,
                zorder=5, marker="*")
    axA.annotate("run in B", xy=(rowB.Q, rowB.RI), xytext=(rowB.Q * 0.60, rowB.RI - 0.44),
                 fontsize=7, color=COL_B, ha="right",
                 arrowprops=dict(arrowstyle="-", lw=0.6, color=COL_B))
    axA.annotate("run in C", xy=(rowC.Q, rowC.RI), xytext=(rowC.Q * 1.7, rowC.RI + 0.05),
                 fontsize=7, color=COL_C,
                 arrowprops=dict(arrowstyle="-", lw=0.6, color=COL_C))
    axA.annotate("Spearman \u03c1 = \u22120.005  (n = 11,938 admissible runs)",
                 xy=(0.02, 0.96), xycoords="axes fraction", ha="left", va="top",
                 fontsize=7, color="0.15")
    axA.set_xlabel("Q-factor (log scale)")
    axA.set_ylabel("Rhythmicity index")
    axA.set_title("Spectral sharpness carries no information about waveform repetition",
                  fontsize=8, loc="left")
    axA.set_xticks([0.3, 1, 3, 10, 30], ["0.3", "1", "3", "10", "30"])
    axA.margins(0.04)
    axA.set_ylim(-0.08, 2.62)

    def trace_row(gsrow, tr, col, ann, ann_va, xlab):
        ax1 = fig.add_subplot(gs[gsrow, 0])
        ax2 = fig.add_subplot(gs[gsrow, 1], sharey=ax1)
        ax1.plot(t, tr, lw=0.5, c=col)
        ax1.axvspan(*DET, color=col, alpha=0.14, lw=0)
        ax1.set_ylim(380, 1020)
        ax1.set_xlim(480, 2020)
        ax1.set_ylabel("Firing nodes")
        ax1.annotate(ann, xy=(0.995, 0.05 if ann_va == "bottom" else 0.97),
                     xycoords="axes fraction", ha="right", va=ann_va,
                     fontsize=7, color="0.1")
        m = (t >= DET[0]) & (t < DET[1])
        ax2.plot(t[m], tr[m], lw=0.9, c=col)
        ax2.set_xlim(DET[0] - 3, DET[1] + 3)
        ax2.set_xticks([DET[0], DET[1]])
        plt.setp(ax2.get_yticklabels(), visible=False)
        ax2.tick_params(axis="y", length=2)
        if xlab:
            ax1.set_xlabel("Timestep")
            ax2.set_xlabel("Timestep (detail)")
        return ax1

    annB = (f"\u03b1 = 0.422, \u03b2 = 0.05\n"
            f"Q = {rowB.Q:.3f} (94th pctile)   RI = {rowB.RI:.3f}")
    annC = (f"\u03b1 = 0.0133, \u03b2 = 0.05\n"
            f"Q = {rowC.Q:.3f}   RI = {rowC.RI:.3f} (sweep max)")
    axB = trace_row(1, trB, COL_B, annB, "bottom", False)
    axC = trace_row(2, trC, COL_C, annC, "top", True)
    axB.set_title("Sharper spectrum, weakly rhythmic waveform (\u03b1 = 0.422)",
                  fontsize=8, loc="left")
    axC.set_title("Ordinary spectrum, rhythmic waveform (identical design except \u03b1 = 0.0133)",
                  fontsize=8, loc="left")

    for ax, L in [(axA, "A"), (axB, "B"), (axC, "C")]:
        bb = ax.get_position()
        fig.text(max(bb.x0 - 0.062, 0.004), min(bb.y1 + 0.012, 0.985), L,
                 fontsize=11, fontweight="bold", va="top")

    fig.savefig(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
