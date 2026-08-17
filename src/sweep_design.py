"""Single source of truth for the sweep design (reviewer item MF-23).

The as-submitted manuscript reported "~250,000 runs" without a grid specification,
replicate count, or seeding scheme, and the number was inconsistent with the
exclusions it also reported. This module fixes the design explicitly so that every
number in the revision is traceable to a reproducible experiment.

DESIGN
------
    networks    5   Erdos-Renyi realisations, N = 20,000, target mean degree K = 200
    locations   5   seed-node choices per network
    alpha      25   log-spaced, 1e-3 to 1e0 inclusive (0.125 decades apart)
    beta       20   linear, 0.05 to 1.00 inclusive (step 0.05)
    replicates 10   independent dynamics seeds per (network, location, alpha, beta)
    -------------------------------------------------------------------------
    total     125,000 runs = 12,500 grid cells x 10 replicates

Each run is 2,500 timesteps; all measurements are taken on the stationary window
[500, 2000) as in the published analysis. Every run has a unique, deterministic
integer seed derived from its position in the design (see `run_seed`), so any single
run can be regenerated in isolation without re-running the sweep.

Network seed convention: net_A uses graph seed 12345, which is the default in the
published repository's `model.generate_er_graph`, so network A reproduces the
published network exactly. Networks B-E use 13345, 14345, 15345, 16345.

Seed-node convention: location 0 is node 0 for every network, which reproduces the
published `init_firing=1` condition. Locations 1-4 are drawn uniformly without
replacement from the remaining nodes using a fixed per-network RNG.
"""
from __future__ import annotations

import numpy as np

# --- network -----------------------------------------------------------------
N_NODES = 20_000
K_DEGREE = 200
NETWORK_NAMES = ["net_A", "net_B", "net_C", "net_D", "net_E"]
GRAPH_SEEDS = [12345, 13345, 14345, 15345, 16345]
N_LOCATIONS = 5
LOCATION_RNG_BASE = 90_000          # rng seed for drawing seed nodes = base + net_idx

# --- parameter grid ----------------------------------------------------------
ALPHAS = np.logspace(-3.0, 0.0, 25)          # 0.001 ... 1.0, 0.125 decades apart
BETAS = np.round(np.arange(0.05, 1.0 + 1e-9, 0.05), 10)   # 0.05 ... 1.00 step 0.05
N_REPLICATES = 10

# --- dynamics / measurement --------------------------------------------------
N_STEPS = 2_500
WINDOW = (500, 2000)                 # stationary measurement window, half-open

N_CELLS = len(NETWORK_NAMES) * N_LOCATIONS * len(ALPHAS) * len(BETAS)
N_RUNS = N_CELLS * N_REPLICATES


def run_seed(net_idx: int, loc_idx: int, a_idx: int, b_idx: int, rep: int) -> int:
    """Unique deterministic RNG seed for one run, from its position in the design."""
    return int(((((net_idx * N_LOCATIONS + loc_idx) * len(ALPHAS) + a_idx)
                 * len(BETAS) + b_idx) * N_REPLICATES) + rep)


def seed_nodes(net_idx: int, n_nodes: int = N_NODES) -> np.ndarray:
    """Seed-node id for each of the `N_LOCATIONS` locations on one network."""
    rng = np.random.default_rng(LOCATION_RNG_BASE + net_idx)
    others = rng.choice(np.arange(1, n_nodes), size=N_LOCATIONS - 1, replace=False)
    return np.concatenate([[0], np.sort(others)]).astype(np.int64)


def design_rows(net_idx: int):
    """Yield (loc_idx, a_idx, b_idx, rep, alpha, beta, seed) for one network."""
    for loc in range(N_LOCATIONS):
        for ai, alpha in enumerate(ALPHAS):
            for bi, beta in enumerate(BETAS):
                for rep in range(N_REPLICATES):
                    yield (loc, ai, bi, rep, float(alpha), float(beta),
                           run_seed(net_idx, loc, ai, bi, rep))


def describe() -> str:
    return (f"{len(NETWORK_NAMES)} networks x {N_LOCATIONS} locations x "
            f"{len(ALPHAS)} alpha x {len(BETAS)} beta x {N_REPLICATES} replicates "
            f"= {N_RUNS:,} runs ({N_CELLS:,} grid cells), {N_STEPS} steps each, "
            f"window [{WINDOW[0]}, {WINDOW[1]}).")
