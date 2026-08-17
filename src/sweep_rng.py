"""Counter-based RNG shared by the CPU and GPU simulators.

Why not just use numpy's or torch's generator. A batched GPU sweep draws randoms for
many runs at once, so a stateful generator makes each run's stream depend on which
other runs happened to share its batch, on the batch size, and on the device. That
would make the sweep unreproducible in exactly the way the reviewer is asking us to
fix (MF-23): a single run could not be regenerated in isolation to check a reported
value.

A counter-based (stateless) RNG removes the dependence. Every uniform variate is a
pure function of (run seed, timestep, node index), so:

  * a run gives identical output whatever batch it lands in, at any batch size;
  * CPU and GPU produce bitwise-identical traces (verified by scripts/00_selftest.py);
  * any single run can be regenerated on its own from its seed alone.

The mixing function is SplitMix64 (Steele, Lea & Flood 2014), applied to the 64-bit
counter ``key * KEY_STRIDE + index``. Uniforms use the top 53 bits, matching the
standard double-precision construction; the simulator compares them against
probabilities in float64 to keep the comparison exact on both devices.

The reference implementation in the published repository draws ONE uniform per node
per timestep and uses it for both the recovery test and the ignition test (a node is
either refractory or resting, so the two tests never apply to the same node in the
same step). This module reproduces that draw structure exactly.
"""
from __future__ import annotations

import numpy as np

MASK64 = (1 << 64) - 1
GOLDEN = 0x9E3779B97F4A7C15
MIX1 = 0xBF58476D1CE4E5B9
MIX2 = 0x94D049BB133111EB
KEY_STRIDE = 0x2545F4914F6CDD1D      # decorrelates streams of adjacent run seeds
TWO_POW_M53 = 1.0 / (1 << 53)


def _as_u64(x):
    """Reinterpret signed int64 as unsigned semantics for numpy (wraparound is fine)."""
    return np.asarray(x, dtype=np.uint64)


def uniform_numpy(key: int, offset: int, n: int) -> np.ndarray:
    """`n` uniforms in [0,1) for stream `key`, counter positions offset .. offset+n-1."""
    idx = np.arange(offset, offset + n, dtype=np.uint64)
    z = (_as_u64(key) * np.uint64(KEY_STRIDE) + idx * np.uint64(GOLDEN))
    z = z & np.uint64(MASK64)
    z ^= (z >> np.uint64(30))
    z = z * np.uint64(MIX1)
    z ^= (z >> np.uint64(27))
    z = z * np.uint64(MIX2)
    z ^= (z >> np.uint64(31))
    return (z >> np.uint64(11)).astype(np.float64) * TWO_POW_M53


def uniform_torch(keys, offset: int, n_nodes: int, device, torch):
    """Uniforms for a BATCH of streams, shape (n_nodes, len(keys)), float64.

    `keys` is a 1-D int64 tensor of run seeds (one per batch column); `offset` is the
    counter position of node 0 for this timestep, i.e. ``step * n_nodes``.
    """
    idx = torch.arange(offset, offset + n_nodes, dtype=torch.int64, device=device).unsqueeze(1)
    z = keys.to(torch.int64).unsqueeze(0) * KEY_STRIDE + idx * GOLDEN
    z = z ^ _lsr(z, 30, torch)
    z = z * MIX1
    z = z ^ _lsr(z, 27, torch)
    z = z * MIX2
    z = z ^ _lsr(z, 31, torch)
    return _lsr(z, 11, torch).to(torch.float64) * TWO_POW_M53


def _lsr(z, k: int, torch):
    """Logical (unsigned) right shift on a signed int64 tensor."""
    return torch.bitwise_right_shift(z, k) & ((1 << (64 - k)) - 1)
