"""Figure 3 — spectral sharpness does not track genuine repetition.

The paper's central simulation result, and the source of Table 1. Two runs from the sweep,
both shown over the identical stationary window (timesteps 500-2000, well beyond the
initial transient), each with a 60-step detail so the cycle-to-cycle waveform is visible:

  a) the single highest-Q run in the entire sweep (alpha = 0.75, beta = 0.10) — the
     sharpest spectral peak anywhere in the dataset, yet the signal is broadband noise
     with no repeating waveform;
  c) the highest-RI run in the sweep (alpha = 0.562341, beta = 1.00) — a lower Q, but a
     waveform that repeats cleanly every cycle.

Ranked by Q the order is a > c; ranked by the rhythmicity index the order reverses.

Every annotated number is MEASURED here at render time by `src.analyze.analyze` on the
plotted window — nothing is hardcoded — so the figure and Table 1 cannot drift apart. The
script prints the Table 1 values it measured.

Panel a's trace is shipped (data/example_traces/fig3A_highestQ.npy) because it comes from
the robustness sweep. Panel c's trace is regenerated from the model seed, which takes
about ten seconds.

Reads   data/example_traces/fig3A_highestQ.npy
Writes  figures/fig3_contrast.png
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import model                      # noqa: E402
from analyze import analyze       # noqa: E402

OUT = Path(__file__).resolve().parent / "fig3_contrast.png"
TRACE_A = ROOT / "data" / "example_traces" / "fig3A_highestQ.npy"

# Measurement window (Methods 2.2 / caption) and the detail inset within it.
WIN = (500, 2000)
ZOOM = (1500, 1560)

# Panel c is regenerated from these exact parameters.
GRAPH = dict(N=20_000, K=200, seed=12345)
RUN_C = dict(alpha=0.562341, beta=1.00, n_steps=2500, init_firing=1, seed=0)
ALPHA_A, BETA_A = 0.75, 0.10   # nominal parameters of the shipped highest-Q run


def apply_style():
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.size": 9,
        "axes.labelsize": 9, "axes.titlesize": 9,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.linewidth": 0.6,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 3, "ytick.major.size": 3,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "figure.dpi": 200,
        "savefig.dpi": 300, "savefig.bbox": "tight",
        "axes.titlelocation": "left", "lines.linewidth": 1.0,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


print("loading panel a (shipped highest-Q trace) ...")
trace_a = np.load(TRACE_A).astype(float)

print(f"regenerating panel c from seed (N={GRAPH['N']}, K={GRAPH['K']}) ...")
adj = model.generate_er_graph(**GRAPH)
trace_c = np.asarray(model.simulate(adj, **RUN_C), dtype=float)

seg_a = trace_a[WIN[0]:WIN[1]]
seg_c = trace_c[WIN[0]:WIN[1]]

# Measure both panels over exactly the window that is plotted.
res_a = analyze(seg_a, fs=1.0)
res_c = analyze(seg_c, fs=1.0)

print("\nTable 1 (measured over timesteps %d-%d):" % WIN)
print(f"  a) highest-Q in sweep : alpha={ALPHA_A:.2f} beta={BETA_A:.2f} "
      f"Q={res_a['Q']:.1f} RI={res_a['RI']:.2f} ({res_a['rhythmicity_class']})")
print(f"  c) highest-RI in sweep: alpha={RUN_C['alpha']:.2f} beta={RUN_C['beta']:.2f} "
      f"Q={res_c['Q']:.1f} RI={res_c['RI']:.2f} ({res_c['rhythmicity_class']})")

apply_style()
panels = [
    ("a", seg_a, res_a, ALPHA_A, BETA_A,
     "Highest-Q run: sharp spectral peak, waveform does not repeat"),
    ("c", seg_c, res_c, RUN_C["alpha"], RUN_C["beta"],
     "Highest-RI run: the waveform repeats every cycle"),
]

fig, axes = plt.subplots(2, 2, figsize=(9.6, 5.6),
                         gridspec_kw={"width_ratios": [3.0, 1.0]})
tt = np.arange(WIN[0], WIN[1])
zoom_mask = (tt >= ZOOM[0]) & (tt < ZOOM[1])

for row, (letter, seg, res, a_val, b_val, title) in enumerate(panels):
    ax_full, ax_zoom = axes[row]

    ax_full.plot(tt, seg, color="#1a1a1a", lw=0.55)
    ax_full.set_xlim(*WIN)
    ax_full.set_ylabel("Firing neurons")
    ax_full.set_title(title, loc="left")
    ax_full.axvspan(ZOOM[0], ZOOM[1], color="#E8A33D", alpha=0.30, zorder=0, lw=0)
    # Q is shown to one decimal: the measured values are 63.5 and 28.3, and rounding
    # them to integers would print "64" where the manuscript states "~63".
    label = (f"$\\alpha$ = {a_val:.2f}    $\\beta$ = {b_val:.2f}\n"
             f"Q $\\approx$ {res['Q']:.1f}   RI = {res['RI']:.2f}")
    ax_full.text(0.015, 0.96, label, transform=ax_full.transAxes,
                 ha="left", va="top", family="monospace", fontsize=7.5,
                 bbox=dict(boxstyle="round,pad=0.32", fc="#f2f2f2", ec="0.7", lw=0.5))
    ax_full.text(-0.075, 1.03, letter, transform=ax_full.transAxes,
                 fontweight="bold", fontsize=11, va="bottom", ha="left")

    ax_zoom.plot(tt[zoom_mask], seg[zoom_mask], color="#1a1a1a", lw=0.9)
    ax_zoom.set_xlim(*ZOOM)
    ax_zoom.set_xticks(list(ZOOM))
    ax_zoom.set_yticks([])
    ax_zoom.spines["left"].set_visible(False)
    ax_zoom.set_title("60-step detail", loc="right", fontsize=7.5, color="0.45")

for ax in axes[-1]:
    ax.set_xlabel("Timestep")

fig.tight_layout(h_pad=1.8, w_pad=1.2)
fig.savefig(OUT, dpi=300, bbox_inches="tight")
print(f"\nwrote {OUT}")
