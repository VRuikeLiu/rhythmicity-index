"""Regenerate the 36 visual-audit traces (bit-exact, counter-based RNG)."""
import os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("risweep"))
from risweep import sim as S, sim_ref, config

sel = pd.read_csv("visual_audit_selection.csv")

def regen_one(args):
    net_name, loc_idx, a_idx, b_idx, rep = args
    net_idx = config.NETWORK_NAMES.index(net_name)
    adj = sim_ref.generate_er_graph(config.N_NODES, config.K_DEGREE,
                                    config.GRAPH_SEEDS[net_idx])
    seed_node = int(config.seed_nodes(net_idx)[loc_idx])
    rs = config.run_seed(net_idx, loc_idx, a_idx, b_idx, rep)
    tr = S.simulate_cpu(adj, float(config.ALPHAS[a_idx]), float(config.BETAS[b_idx]),
                        config.N_STEPS, seed_node, rs)
    return tr

if __name__ == "__main__":
    from multiprocessing import Pool
    jobs = [(r.network, int(r.location_idx), int(r.a_idx), int(r.b_idx), int(r.replicate))
            for r in sel.itertuples()]
    t0 = time.time()
    with Pool(9) as pool:
        traces = pool.map(regen_one, jobs)
    arr = np.stack(traces)
    np.savez_compressed("visual_audit_traces.npz", traces=arr)
    # verify mean activity against the delivered table
    w = arr[:, 500:2000].astype(float)
    ma = w.mean(axis=1)
    err = np.abs(ma - sel.mean_activity.values)
    print("regen %.0fs; max |mean_act - table| = %.6f" % (time.time() - t0, err.max()))
