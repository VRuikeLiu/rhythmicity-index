"""Neural cascade model: Erdos-Renyi graph + Greenberg-Hastings dynamics.

This is the "controlled signal generator" used throughout the study. It builds a
random network and runs a 3-state Greenberg-Hastings cascade whose activity time
series A(t) (number of firing nodes per step) is the signal fed to the
rhythmicity index (see rhythmicity.py).

State codes: 0 = Resting, 1 = Firing, 2 = Refractory.

Update rule (Model A), applied synchronously each step:
  * Firing    -> Refractory      (deterministic)
  * Refractory-> Resting         (stochastic, probability beta)
  * Resting   -> Firing          (stochastic, probability P_fire below)
where, for a resting node with m firing neighbours,
      P_fire = 1 - (1 - alpha)**m        (independent-neighbour "OR" rule).

Reproducibility: the network is rebuilt from a fixed integer seed, so no graph
file needs to be shipped. Record (N, K, seed) to reproduce a network exactly.
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp

RESTING, FIRING, REFRACTORY = 0, 1, 2

# Study defaults
N_DEFAULT = 20_000    # number of nodes
K_DEFAULT = 200       # target average degree
GRAPH_SEED_DEFAULT = 12345


def generate_er_graph(N: int = N_DEFAULT, K: int = K_DEFAULT,
                      seed: int = GRAPH_SEED_DEFAULT) -> sp.csr_matrix:
    """Build an undirected Erdos-Renyi graph (no self-loops) as a CSR matrix.

    Edge probability p = K / (N - 1) gives target mean degree K. The upper
    triangle is drawn, then symmetrised.
    """
    rng = np.random.default_rng(seed)
    p = K / (N - 1)
    rows, cols = [], []
    for i in range(N):
        n_possible = N - i - 1
        if n_possible <= 0:
            continue
        n_edges = rng.binomial(n_possible, p)
        if n_edges:
            targets = rng.choice(n_possible, size=n_edges, replace=False) + i + 1
            rows.extend([i] * n_edges)
            cols.extend(targets.tolist())
    data = np.ones(len(rows), dtype=np.float32)
    upper = sp.csr_matrix((data, (np.array(rows, np.int32), np.array(cols, np.int32))),
                          shape=(N, N))
    adj = upper + upper.T
    adj.data[:] = 1.0
    return adj


def simulate(adj: sp.csr_matrix, alpha: float, beta: float, n_steps: int,
             init_firing: np.ndarray | int = 1, seed: int = 0) -> np.ndarray:
    """Run one Greenberg-Hastings cascade and return the activity series A(t).

    Parameters
    ----------
    adj : CSR adjacency (N x N), symmetric, binary.
    alpha : ignition probability per firing neighbour.
    beta : per-step refractory -> resting recovery probability.
    n_steps : number of timesteps.
    init_firing : indices of initially-firing nodes, or an int k to seed the
        first k nodes (k=1 reproduces the "n1_fixed" condition).
    seed : RNG seed for this run.

    Returns
    -------
    A_t : int array, shape (n_steps,) -- firing-node count at each step. The run
        terminates early (zeros thereafter) if activity dies out.
    """
    N = adj.shape[0]
    rng = np.random.default_rng(seed)
    state = np.zeros(N, dtype=np.uint8)
    if np.isscalar(init_firing):
        state[np.arange(int(init_firing))] = FIRING
    else:
        state[np.asarray(init_firing, dtype=np.int64)] = FIRING

    A_t = np.zeros(n_steps, dtype=np.int32)
    for t in range(n_steps):
        firing = (state == FIRING).astype(np.float32)
        if firing.sum() == 0:
            break
        m = adj.dot(firing)                       # firing-neighbour count per node
        P_fire = 1.0 - np.power(1.0 - alpha, m)   # OR-rule ignition probability
        r = rng.random(N, dtype=np.float32)
        new = state.copy()
        new[state == FIRING] = REFRACTORY                              # deterministic
        new[(state == REFRACTORY) & (r < beta)] = RESTING             # stochastic recovery
        ign = (state == RESTING) & (r < P_fire)
        new[ign] = FIRING                                            # stochastic ignition
        state = new
        A_t[t] = int((state == FIRING).sum())
    return A_t


if __name__ == "__main__":
    # Tiny smoke test: build a network and show a short activity series.
    adj = generate_er_graph(N=2000, K=200, seed=GRAPH_SEED_DEFAULT)
    a = simulate(adj, alpha=0.01, beta=0.05, n_steps=500, init_firing=1, seed=0)
    print("mean activity:", a[a > 0].mean() if (a > 0).any() else 0.0,
          "| survived:", bool(a[-1] > 0))
