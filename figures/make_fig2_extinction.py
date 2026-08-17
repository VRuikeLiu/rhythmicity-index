"""Figure 2 -- the extinct-active transition: sharp, at the mean-field prediction,
and independent of the recovery probability beta.

Two panels, both computed from results/per_cell_summary.csv (the locked-spec sweep:
every (alpha, beta) grid cell pools 250 runs = 5 ER network realisations x 5 seed-node
locations x 10 replicates).

A) Survival probability vs alpha (log scale) at five beta values spanning the sweep
   (0.05, 0.25, 0.50, 0.75, 1.00), with 95% Wilson binomial CIs. The five curves are
   slightly offset horizontally so overlapping intervals stay visible. Dotted line:
   mean-field prediction alpha_C ~ 1/K = 0.005 (Kinouchi & Copelli 2006).

B) Critical point fitted separately at each of the 20 beta values by maximum-likelihood
   fit of the mean-field branching-process survival probability -- s solving
   s = 1 - exp(-(alpha/alpha_C) s) -- to the 8 grid points spanning the transition;
   error bars are 95% profile-likelihood intervals. Pooled over beta:
   alpha_C = (5.03 +/- 0.03) x 10^-3, within 0.6% of 1/K. The old manuscript value
   0.0042 was the last fully-extinct grid value: a lower bound set by the 0.125-decade
   grid spacing, not an estimate of the critical point.

Reads   results/per_cell_summary.csv
Writes  figures/fig2_extinction.png, results/fig2_alphaC_fits.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import chi2, norm

ROOT = Path(__file__).resolve().parent.parent
CELLS = ROOT / "results" / "per_cell_summary.csv"
OUT_PNG = Path(__file__).resolve().parent / "fig2_extinction.png"
OUT_CSV = ROOT / "results" / "fig2_alphaC_fits.csv"

df = pd.read_csv(CELLS)
alphas = np.sort(df.alpha.unique())
betas = np.sort(df.beta.unique())
FIT_ALPHAS = alphas[5:13]  # 4.2e-3 .. 3.2e-2, the 8 points spanning the transition


def wilson(k, n, z=norm.ppf(0.975)):
    p = k / n
    den = 1 + z**2 / n
    ctr = (p + z**2 / (2 * n)) / den
    hw = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return ctr - hw, ctr + hw


def surv_branching(alpha, alpha_c):
    """Survival probability of a mean-field branching process, offspring mean alpha/alpha_c."""
    sig = alpha / alpha_c
    if sig <= 1:
        return 0.0
    return brentq(lambda s: s - (1 - np.exp(-sig * s)), 1e-12, 1, xtol=1e-14)


def fit_alpha_c(k, n, bounds=(0.004, 0.006)):
    """ML alpha_C + 95% profile-likelihood CI from binomial counts on FIT_ALPHAS."""
    def negll(ac):
        ll = 0.0
        for ai, ki, ni in zip(FIT_ALPHAS, k, n):
            p = min(max(surv_branching(ai, ac), 1e-12), 1 - 1e-12)
            ll += ki * np.log(p) + (ni - ki) * np.log(1 - p)
        return -ll

    r = minimize_scalar(negll, bounds=bounds, method="bounded")
    ac, ll0, thr = r.x, -r.fun, chi2.ppf(0.95, 1) / 2
    lo = brentq(lambda a: (negll(a) + ll0) - thr, bounds[0], ac)
    hi = brentq(lambda a: (negll(a) + ll0) - thr, ac, bounds[1])
    return ac, lo, hi


cells = df.set_index(["alpha", "beta"])
pooled = df.groupby("alpha").agg(n=("n_runs", "sum"), k=("n_surviving", "sum")).loc[FIT_ALPHAS]
ac_pool, lo_pool, hi_pool = fit_alpha_c(pooled.k.values, pooled.n.values)

per_beta = {
    b: fit_alpha_c(
        np.array([cells.loc[(a, b), "n_surviving"] for a in FIT_ALPHAS]),
        np.array([cells.loc[(a, b), "n_runs"] for a in FIT_ALPHAS]),
    )
    for b in betas
}
pb = pd.DataFrame(per_beta, index=["alpha_c_ml", "ci95_lo", "ci95_hi"]).T
pb.index.name = "beta"

fits_out = pd.concat(
    [pb.reset_index().assign(scope="per_beta"),
     pd.DataFrame([{"beta": np.nan, "alpha_c_ml": ac_pool, "ci95_lo": lo_pool,
                    "ci95_hi": hi_pool, "scope": "pooled"}])],
    ignore_index=True,
)[["scope", "beta", "alpha_c_ml", "ci95_lo", "ci95_hi"]]
fits_out.to_csv(OUT_CSV, index=False)

# sanity: the fit must reproduce the paper's quoted values
assert abs(ac_pool - 5.03e-3) < 0.01e-3, ac_pool
assert max(abs(pb.alpha_c_ml - 5e-3)) / 5e-3 < 0.03

# ---------------------------------------------------------------- figure
SHOW_BETAS = [0.05, 0.25, 0.50, 0.75, 1.00]
cmap = mpl.cm.viridis
cols = {b: cmap(0.10 + 0.70 * i / 4) for i, b in enumerate(SHOW_BETAS)}

mpl.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                     "xtick.labelsize": 6, "ytick.labelsize": 6,
                     "axes.spines.top": False, "axes.spines.right": False})

fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.2, 3.0), dpi=300,
                               gridspec_kw=dict(width_ratios=[1.55, 1.0], wspace=0.32))

for i, b in enumerate(SHOW_BETAS):
    s = df[df.beta == b].sort_values("alpha")
    x = np.maximum(s.alpha.values * (1 + 0.032 * (i - 2)), 1e-3)
    y = s.survival.values
    lo, hi = wilson(s.n_surviving.values, s.n_runs.values)
    axA.errorbar(x, y, yerr=np.clip(np.vstack([y - lo, hi - y]), 0, None), fmt="o-", ms=2.6, lw=0.9,
                 capsize=1.4, elinewidth=0.7, color=cols[b], label=f"$\\beta$ = {b:.2f}",
                 zorder=3)

axA.set_xscale("log")
axA.axvline(0.005, color="0.35", lw=0.8, ls=":", zorder=1)
axA.text(0.005 * 0.82, 0.42, "mean-field\nprediction\n$1/K = 0.005$", ha="right",
         va="center", fontsize=6, color="0.25")
axA.set_xlabel("$\\alpha$ (per-edge firing probability)")
axA.set_ylabel("Survival probability")
axA.set_title("The extinction\u2013survival transition\ndoes not move with $\\beta$",
              fontsize=8, loc="left")
axA.legend(frameon=False, fontsize=6, loc="center left", bbox_to_anchor=(0.02, 0.72),
           handlelength=1.4, labelspacing=0.25, borderaxespad=0.0)
axA.set_ylim(-0.045, 1.06)
axA.margins(x=0.05)

yb = pb.alpha_c_ml.values * 1e3
axB.errorbar(pb.index.values, yb,
             yerr=[(pb.alpha_c_ml - pb.ci95_lo).values * 1e3,
                   (pb.ci95_hi - pb.alpha_c_ml).values * 1e3],
             fmt="o", ms=2.8, capsize=1.6, elinewidth=0.7, color="0.15", zorder=3)
axB.axhline(5.0, color="0.35", lw=0.8, ls=":", zorder=1)
axB.text(1.01, 5.0, "$1/K$", ha="left", va="center", fontsize=6, color="0.25",
         transform=mpl.transforms.blended_transform_factory(axB.transAxes, axB.transData))
axB.set_xlabel("$\\beta$ (recovery probability)")
axB.set_ylabel("Fitted critical point $\\alpha_C$ ($\\times10^{-3}$)")
axB.set_title("Fitted $\\alpha_C$ shows no trend in $\\beta$\nand stays within 3% of $1/K$",
              fontsize=8, loc="left")
ylo_, yhi_ = pb.ci95_lo.min() * 1e3, pb.ci95_hi.max() * 1e3
pad_ = 0.06 * (yhi_ - ylo_)
axB.set_ylim(ylo_ - pad_, yhi_ + pad_)
axB.margins(x=0.06)

for ax, letter in [(axA, "A"), (axB, "B")]:
    ax.text(-0.14 if ax is axB else -0.10, 1.08, letter, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top", ha="left")

fig.savefig(OUT_PNG, bbox_inches="tight")
print(f"alpha_C (pooled) = {ac_pool:.6f}  CI [{lo_pool:.6f}, {hi_pool:.6f}]")
print(f"wrote {OUT_PNG.name}, {OUT_CSV.name}")
