# Rhythmicity Index

Reference implementation and reproducibility materials for the **Rhythmicity Index (RI)** —
a measure of whether a neural activity time series is *genuinely periodic* (a repeating
waveform), as opposed to merely having a sharp spectral peak. From the project
*"Neural cascade oscillation study"* by Ruike (Vincent) Liu.

A reviewer can (a) read the method and confirm it matches the paper, and (b) regenerate
the figures from small processed data files — **without** rerunning 250,000 simulations or
downloading multi-GB raw traces.

## Repository layout

```
rhythmicity-index/
├── README.md              — this file
├── METHODS_NOTES.md       — exact RI formula, EEG dataset identity, threshold usage, the 2.2% derivation
├── requirements.txt
├── LICENSE                — MIT
├── src/
│   ├── rhythmicity.py     — THE core deliverable: lagged coherence + cycle consistency + compute_rhythmicity_index + classify_rhythmicity (canonical, uncapped)
│   ├── model.py           — Erdos–Renyi graph + Greenberg–Hastings cascade (the "controlled signal generator")
│   └── eeg_validation.py  — EEGMMIDB loader (mne.datasets.eegbci) + per-epoch RI (same RI applied to real EEG)
├── figures/
│   ├── make_fig2_contrast.py   — Fig 2: Q-factor vs. true periodicity (3-panel)
│   ├── make_fig4_extinction.py — Fig 4: survival/extinction threshold vs. alpha
│   ├── make_fig_eeg.py         — Fig 3: EEG eyes-open vs eyes-closed RI
│   ├── CAPTIONS.md             — exact (alpha, beta, network, replicate) + canonical RI for every panel
│   └── *.png                   — rendered figures (300 dpi)
└── data/
    ├── run_summary.csv         — one row per (network, location, alpha, beta) gridpoint:
    │                             extinction_prob, mean_activity, amplitude, period, Q, RI (canonical), class
    ├── example_traces/         — the 3 activity traces behind Fig 2 (.npy and .csv)
    └── eeg_rhythmicity.csv      — per-subject, per-condition mean RI (20 subjects × {EO, EC})
```

## The rhythmicity index, in one paragraph

RI combines two scores computed on the post-burn-in activity window `A(t)`:
**phase consistency** (lagged coherence; Fransen et al. 2015) and **shape consistency**
(cycle-to-cycle Pearson correlation; Cole & Voytek 2019). They are combined by a
**min-of-two-gates (logical AND) rule** mapped onto three bands — `RI ≥ 2.0` RHYTHMIC,
`1.0 ≤ RI < 2.0` WEAKLY_RHYTHMIC, `RI < 1.0` ARRHYTHMIC — so a run scores high only when
phase *and* shape are both strong (a sharp spectral peak alone is not enough). The exact
formula, thresholds, and a compact equation are in **[METHODS_NOTES.md](METHODS_NOTES.md)**
and implemented verbatim in [`src/rhythmicity.py`](src/rhythmicity.py).

## Reproduce the figures

```bash
pip install -r requirements.txt
python figures/make_fig2_contrast.py     # Fig 2  (reads data/example_traces/)
python figures/make_fig4_extinction.py   # Fig 4  (reads data/run_summary.csv)
python figures/make_fig_eeg.py           # Fig 3  (reads data/eeg_rhythmicity.csv)
```

To rebuild a network and generate a fresh activity series from scratch:

```python
from src.model import generate_er_graph, simulate
adj = generate_er_graph(N=20_000, K=200, seed=12345)
A_t = simulate(adj, alpha=0.010, beta=0.05, n_steps=2000, init_firing=1, seed=0)
```

## Data & code availability

- **Human EEG:** EEG Motor Movement/Imagery Database (EEGMMIDB), PhysioNet
  (Schalk et al. 2004; Goldberger et al. 2000),
  https://physionet.org/content/eegmmidb/1.0.0/. Downloads automatically via
  `mne.datasets.eegbci`; no EDF files are committed here.
- **Simulation data:** the processed `data/run_summary.csv` and the three example traces are
  included. Full raw per-run traces (≈250k runs) are not committed; `src/model.py` rebuilds
  any network from its seed.

## Citation

> Liu, R. V. *Neural cascade oscillation study* (2026). Rhythmicity Index code and data,
> https://github.com/VRuikeLiu/rhythmicity-index

## License

MIT — see [LICENSE](LICENSE).
