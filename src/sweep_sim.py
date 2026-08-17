"""CPU regeneration of individual sweep runs (counter-based RNG).

Any single run of the 125,000-run sweep is regenerable in isolation:
its random stream is a pure function of (run_seed, timestep, node index)
via the SplitMix64 counter RNG in `sweep_rng.py` (see `sweep_design.py`
for the seed conventions). Regenerated traces are bitwise-identical to
the sweep's stored traces at any batch size, on CPU or GPU.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

import sweep_rng as _rng

RESTING, FIRING, REFRACTORY = 0, 1, 2

def generate_er_graph(N: int, K: int, seed: int) -> sp.csr_matrix:
    """Undirected Erdos-Renyi graph as CSR. Identical to the published construction."""
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


def simulate_cpu(adj: sp.csr_matrix, alpha: float, beta: float, n_steps: int,
                 seed_node: int, run_seed: int) -> np.ndarray:
    """One run on the CPU using the counter-based RNG. Returns A(t), int32."""
    N = adj.shape[0]
    state = np.zeros(N, dtype=np.uint8)
    state[seed_node] = FIRING
    A_t = np.zeros(n_steps, dtype=np.int32)
    log1m = np.log1p(-alpha) if alpha < 1.0 else -np.inf
    for t in range(n_steps):
        firing = (state == FIRING)
        if not firing.any():
            break
        m = adj.dot(firing.astype(np.float32))
        if alpha < 1.0:
            P_fire = 1.0 - np.exp(log1m * m.astype(np.float64))
        else:
            P_fire = (m > 0).astype(np.float64)
        r = _rng.uniform_numpy(run_seed, t * N, N)
        new = state.copy()
        new[firing] = REFRACTORY
        new[(state == REFRACTORY) & (r < beta)] = RESTING
        new[(state == RESTING) & (r < P_fire)] = FIRING
        state = new
        A_t[t] = int((state == FIRING).sum())
    return A_t
