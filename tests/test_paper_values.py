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
def trace_a():
    return np.load(ROOT / "data" / "example_traces" / "fig3A_highestQ.npy").astype(float)


# --- Table 1 / Figure 3 -----------------------------------------------------------

def test_panel_a_matches_table1(trace_a):
    """Highest-Q run: Q ~ 63, RI = 1.62, and it is NOT classified rhythmic."""
    res = analyze(trace_a[WIN[0]:WIN[1]], fs=1.0)
    assert res["Q"] == pytest.approx(63.5, abs=0.5)
    assert res["RI"] == pytest.approx(1.62, abs=0.01)
    assert res["rhythmicity_class"] == "WEAKLY_RHYTHMIC"


@pytest.mark.slow
def test_panel_c_matches_table1():
    """Highest-RI run, regenerated from seed: Q ~ 28, RI = 3.49, RHYTHMIC."""
    import model
    adj = model.generate_er_graph(N=20_000, K=200, seed=12345)
    trace = np.asarray(model.simulate(adj, alpha=0.562341, beta=1.00,
                                      n_steps=2500, init_firing=1, seed=0), dtype=float)
    res = analyze(trace[WIN[0]:WIN[1]], fs=1.0)
    assert res["Q"] == pytest.approx(28.3, abs=0.5)
    assert res["RI"] == pytest.approx(3.49, abs=0.02)
    assert res["rhythmicity_class"] == "RHYTHMIC"


@pytest.mark.slow
def test_q_and_ri_rank_the_two_runs_oppositely(trace_a):
    """The paper's central claim: Q ranks a > c, the rhythmicity index ranks c > a."""
    import model
    adj = model.generate_er_graph(N=20_000, K=200, seed=12345)
    trace_c = np.asarray(model.simulate(adj, alpha=0.562341, beta=1.00,
                                        n_steps=2500, init_firing=1, seed=0), dtype=float)
    a = analyze(trace_a[WIN[0]:WIN[1]], fs=1.0)
    c = analyze(trace_c[WIN[0]:WIN[1]], fs=1.0)
    assert a["Q"] > c["Q"]
    assert c["RI"] > a["RI"]


# --- Q-factor convention ----------------------------------------------------------

def test_q_depends_on_nperseg(trace_a):
    """Q is resolution-dependent; the paper's value requires nperseg=256."""
    seg = trace_a[WIN[0]:WIN[1]]
    assert compute_q_factor(seg, nperseg=256)["Q"] == pytest.approx(63.5, abs=0.5)
    assert compute_q_factor(seg, nperseg=512)["Q"] > 70.0


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
    """alpha_c = 0.0042 is the last fully-extinct alpha; alpha = 1 re-extinguishes."""
    df = pd.read_csv(ROOT / "data" / "run_summary.csv")
    sub = df[(df["network"] == "net_A") & (df["location_idx"] == 0)
             & np.isclose(df["beta"], 0.05)].sort_values("alpha")
    alpha = sub["alpha"].to_numpy()
    survival = 1.0 - sub["extinction_prob"].to_numpy()
    # alpha = 1 is the degenerate deterministic limit and is also fully extinct, so the
    # transition is located below it.
    below = alpha < 1.0
    last_extinct = alpha[below][survival[below] == 0.0].max()
    assert last_extinct == pytest.approx(0.0042, abs=1e-4)
    # activity self-extinguishes again at the alpha = 1 endpoint
    assert survival[np.isclose(alpha, 1.0)][0] == 0.0


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
