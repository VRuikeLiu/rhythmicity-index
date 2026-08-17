"""Regression tests pinning the numbers the paper reports.

Run with:  pytest -q          (or:  python tests/test_paper_values.py)

These lock the published values to the code. If a change to rhythmicity.py, spectral.py or
the Welch convention moves any of them, these tests fail rather than letting the repo drift
quietly out of agreement with the manuscript.

The panel-c test regenerates a 20,000-node network and a 2,500-step run (~10 s). It is
marked `slow`; deselect with:  pytest -q -m "not slow"
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from analyze import analyze              # noqa: E402
from spectral import compute_q_factor    # noqa: E402
from rhythmicity import compute_rhythmicity_index  # noqa: E402

WIN = (500, 2000)


@pytest.fixture(scope="module")
def trace_b():
    return np.load(ROOT / "data" / "example_traces" / "fig3B_sharpQ.npy").astype(float)


@pytest.fixture(scope="module")
def trace_c():
    return np.load(ROOT / "data" / "example_traces" / "fig3C_highestRI.npy").astype(float)


# --- Table 1 / Figure 3 -----------------------------------------------------------
# Panel values are pinned to the sweep delivery table (results/per_run_results.csv.gz),
# the paper's record. Q, c and the period regenerate bit-for-bit across machines; the
# shape term r (hence RI) can differ in the last digits across SciPy builds, so RI
# tolerances are looser (see README "Reproducibility notes").

def test_panel_b_matches_table1(trace_b):
    """Sharp-spectrum comparison run: Q = 12.491 (not resolution-limited), weakly rhythmic."""
    from rhythmicity_locked_freq import analyze_s2
    res = analyze_s2(trace_b[WIN[0]:WIN[1]], fs=1.0)
    assert res["Q"] == pytest.approx(12.491, abs=0.001)
    assert not res["Q_resolution_limited"]
    assert res["c"] == pytest.approx(0.229168, abs=1e-4)
    assert res["period"] == pytest.approx(12.0887, abs=0.001)
    assert res["RI"] == pytest.approx(1.342, abs=0.05)
    assert res["label"] == "WEAKLY_RHYTHMIC"


def test_panel_c_matches_table1(trace_c):
    """Highest-RI admissible run: RI = 2.378 at period 26.08, rhythmic, ordinary Q."""
    from rhythmicity_locked_freq import analyze_s2
    res = analyze_s2(trace_c[WIN[0]:WIN[1]], fs=1.0)
    assert res["Q"] == pytest.approx(2.398, abs=0.001)
    assert res["period"] == pytest.approx(26.081, abs=0.001)
    assert res["samples_per_cycle"] >= 8 and res["n_cycles"] >= 10
    assert res["RI"] == pytest.approx(2.378, abs=0.02)
    assert res["label"] == "RHYTHMIC"


def test_q_and_ri_rank_the_two_runs_oppositely(trace_b, trace_c):
    """The paper's contrast: Q ranks B > C, the rhythmicity index ranks C > B."""
    from rhythmicity_locked_freq import analyze_s2
    b = analyze_s2(trace_b[WIN[0]:WIN[1]], fs=1.0)
    c = analyze_s2(trace_c[WIN[0]:WIN[1]], fs=1.0)
    assert b["Q"] > c["Q"]
    assert c["RI"] > b["RI"]


@pytest.mark.slow
def test_figure3_traces_regenerate_from_seed(trace_b, trace_c):
    """Both shipped traces regenerate bitwise from their design-position seeds."""
    import sweep_sim, sweep_design as D
    adj = sweep_sim.generate_er_graph(D.N_NODES, D.K_DEGREE, D.GRAPH_SEEDS[2])  # net_C
    node = int(D.seed_nodes(2)[0])
    for seed, alpha, shipped in ((54202, 0.42169650342858224, trace_b),
                                 (51802, 0.013335214321633242, trace_c)):
        tr = sweep_sim.simulate_cpu(adj, alpha, 0.05, 2500, node, seed)
        assert np.array_equal(tr, shipped.astype(tr.dtype))


# --- Figure 1 / concept schematic -------------------------------------------------

def test_fig1_q_is_identical_across_panels():
    """Fig 1's premise: all three schematic traces have the same Q (8.5), only shape differs.

    Requires nperseg=2048 — these traces have a 120-sample fundamental, and the paper's
    default 256 (suited to the 2-6 timestep periods of the simulation traces) cannot
    resolve the peak and collapses Q to 1.0 for all three.
    """
    sys.path.insert(0, str(ROOT / "figures"))
    from make_fig1_concept import build_trace, NPERSEG

    results = {k: analyze(build_trace(4, mix), fs=1.0, nperseg=NPERSEG)
               for k, mix in (("a", 0.00), ("b", 0.55), ("c", 1.00))}

    q = [results[k]["Q"] for k in "abc"]
    assert all(v == pytest.approx(8.5, abs=0.01) for v in q), q

    # phase coherence is ~1 in every panel; only shape consistency and RI move
    assert all(results[k]["lagged_coherence"] == pytest.approx(1.0, abs=0.01) for k in "abc")
    shape = [results[k]["mean_cycle_corr"] for k in "abc"]
    ri = [results[k]["RI"] for k in "abc"]
    assert shape == sorted(shape), shape
    assert ri == sorted(ri), ri
    assert ri[0] < 2.0 <= ri[1] < ri[2]


# --- Q-factor convention ----------------------------------------------------------

def test_q_depends_on_nperseg(trace_b):
    """Q is resolution-dependent; the paper's convention is nperseg=256 (locked spec)."""
    from rhythmicity_locked import q_factor
    seg = trace_b[WIN[0]:WIN[1]]
    at256 = q_factor(seg, fs=1.0, nperseg=256)
    at512 = q_factor(seg, fs=1.0, nperseg=512)
    assert at256["Q"] == pytest.approx(12.491, abs=0.001)
    assert not at256["resolution_limited"]
    assert at512["Q"] > 70.0 and at512["resolution_limited"] and at512["undersegmented"]


# --- Figure 4 / EEG validation ----------------------------------------------------

def test_eeg_berger_effect():
    """EO mean 0.57, EC mean 1.52, 19/20 subjects increase (Figure 4 caption)."""
    df = pd.read_csv(ROOT / "data" / "eeg_rhythmicity.csv")
    wide = df.pivot(index="subject", columns="condition", values="mean_RI")
    assert len(wide) == 20
    assert int(df["n_epochs"].sum()) == 600
    assert wide["EO"].mean() == pytest.approx(0.57, abs=0.01)
    assert wide["EC"].mean() == pytest.approx(1.52, abs=0.01)
    assert int((wide["EC"] - wide["EO"] > 0).sum()) == 19


# --- Figure 2 / extinction transition ---------------------------------------------

def test_extinction_threshold():
    """Zero-survivor floor at alpha <= 0.0042; fitted alpha_C = (5.03 +/- 0.03) x 10^-3;
    alpha = 1 re-extinguishes. (Revision: alpha_C is now estimated by the
    branching-process ML fit in figures/make_fig2_extinction.py, not quoted as the
    last fully-extinct grid value, which is only a grid-spacing lower bound.)"""
    df = pd.read_csv(ROOT / "results" / "per_cell_summary.csv")
    pooled = df.groupby("alpha").agg(n=("n_runs", "sum"), k=("n_surviving", "sum"))
    alpha = pooled.index.to_numpy()
    # zero-survivor floor: no run survives at or below 0.0042 (30,000 runs)
    low = pooled[alpha <= 0.00422]
    assert low.n.sum() == 30000 and low.k.sum() == 0
    # survival onset at the next grid value
    onset = pooled.loc[alpha[alpha > 0.00422].min()]
    assert onset.k / onset.n == pytest.approx(0.2026, abs=1e-4)
    # the paper's fitted critical point (values written by make_fig2_extinction.py)
    fits = pd.read_csv(ROOT / "results" / "fig2_alphaC_fits.csv")
    pooled_fit = fits[fits.scope == "pooled"].iloc[0]
    assert pooled_fit.alpha_c_ml == pytest.approx(5.03e-3, abs=0.01e-3)
    assert pooled_fit.ci95_lo == pytest.approx(5.00e-3, abs=0.01e-3)
    assert pooled_fit.ci95_hi == pytest.approx(5.06e-3, abs=0.01e-3)
    # per-beta fits: all within 3% of 1/K = 0.005 (beta-independence)
    per_beta = fits[fits.scope == "per_beta"]
    assert len(per_beta) == 20
    assert (np.abs(per_beta.alpha_c_ml - 5e-3) / 5e-3).max() < 0.03
    # activity self-extinguishes again at the alpha = 1 endpoint (all 5,000 runs)
    top = pooled.loc[1.0]
    assert top.n == 5000 and top.k == 0


# --- Index structure: the AND-across-criteria rule --------------------------------

def test_and_rule_blocks_single_criterion():
    """Perfect phase with poor shape (or the reverse) cannot reach the rhythmic band."""
    phase_only = compute_rhythmicity_index(1.0, 1.0, 0.05)
    shape_only = compute_rhythmicity_index(0.05, 0.05, 1.0)
    assert phase_only < 2.0
    assert shape_only < 2.0
    assert compute_rhythmicity_index(1.0, 1.0, 1.0) >= 2.0


def test_index_is_monotone_in_shape():
    """With phase held perfect, the index rises with cycle correlation."""
    scores = [compute_rhythmicity_index(1.0, 1.0, r) for r in (0.0, 0.2, 0.4, 0.7, 1.0)]
    assert scores == sorted(scores)


def test_minimal_evidence_floor():
    """Values at or below the 0.1 floor contribute no evidence."""
    assert compute_rhythmicity_index(0.1, 0.1, 0.1) < 1.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
