# Rhythmicity Index

Reference implementation and reproducibility materials for the paper

> **Rhythmicity Index: Signifying Shape Coherence in Addition to Phase Coherence**

A signal with a sharp spectral peak is routinely called an oscillation. This repository
contains the code behind a demonstration that spectral sharpness alone does not establish
that a waveform actually *repeats* — and a **rhythmicity index (RI)** that tests periodicity
in the time domain instead, by requiring phase consistency **and** shape consistency at the
same time.

The headline result: across 125,000 simulated runs (11,938 admissible under the
estimator's sampling floors), Q and RI are uncorrelated — Spearman ρ = −0.005 — and
every run in the top percentile of Q falls below the rhythmic threshold. A matched
pair makes it concrete: two runs identical in design except the firing probability α
order one way by Q (12.49 vs 2.40) and the opposite way by RI (1.342 vs 2.378). The
same index, developed entirely on simulated signals and applied unchanged to human
EEG, recovers the Berger effect.

Everything the paper reports about individual signals is reproducible here from small
processed files — **no rerunning of 125,000 simulations and no multi-GB downloads.**

## Quickstart

```bash
pip install -r requirements.txt
pytest -q                       # verify the published numbers reproduce (~18 s)
python figures/make_fig1_concept.py
python figures/make_fig2_extinction.py
python figures/make_fig3_contrast.py    # regenerates its trace from seed (~10 s)
python figures/make_fig4_eeg.py
```

Score any signal of your own with both measures:

```python
import sys; sys.path.insert(0, "src")
from analyze import analyze

res = analyze(my_signal, fs=1.0)        # fs=160.0 for the EEG recordings
print(res["Q"], res["RI"], res["rhythmicity_class"])
```

## The rhythmicity index

RI is computed from three quantities measured on the activity time series `A(t)`:

| Symbol | Quantity | Reference |
|---|---|---|
| `c` | three-cycle lagged coherence — phase consistency, windows exactly 3 cycles long | Fransen et al. 2015 |
| `c̄` | mean lagged coherence — the same measure averaged over window lengths 1…10 cycles | Fransen et al. 2015 |
| `r` | mean cycle correlation — Pearson `r` between successive peak/trough-aligned, resampled cycles | Cole & Voytek 2019 |

Each is divided by a threshold to give a "fraction of the passing bar", then combined:

```
Φ  = max(c/0.5, c̄/0.35)          Ψ  = r/0.4              # phase / shape evidence
g_s = min(Φ, Ψ)                                           # strong gate  (AND)

Φ_w = max(c/0.3, c̄/0.25)         Ψ_w = r/0.2              # looser thresholds
Φ_0 = max(c, c̄)/0.1              Ψ_0 = r/0.1              # minimal evidence, 0 if numerator <= 0.1
g_w = max( min(Φ_w, Ψ_0), min(Ψ_w, Φ_0) )                 # intermediate gate

RI  =  2 + (g_s - 1)                     if g_s >= 1       # RHYTHMIC        (RI >= 2)
       min(1 + (g_w - 1), 1.999)         elif g_w >= 1     # WEAKLY_RHYTHMIC (1 <= RI < 2)
       min(max(g_s, g_w), 0.999)         otherwise         # ARRHYTHMIC      (RI < 1)
```

The structure is **OR within a criterion, AND across criteria**. The `max` in `Φ` is
deliberately lenient: the two coherence estimators detect regularity at different timescales
and fail in opposite ways, so a signal is credited with phase consistency if it is regular at
*either* timescale. All of the strictness lives in the `min` of `g_s` — the AND across phase
and shape — which is what prevents a spectrally sharp but non-repeating signal from scoring
highly. The thresholds are reasonable settings, not tuned constants; what the paper's results
rely on is the *ordering* the index produces.

Implemented in [`src/rhythmicity.py`](src/rhythmicity.py) (`compute_rhythmicity_index`,
`classify_rhythmicity`).

### A note on Q

Q is resolution-dependent, so the estimator settings are part of the measurement. The paper
uses Welch with `nperseg = min(256, len(signal))` and takes the half-power bandwidth from the
outermost bins above half peak power. **Changing `nperseg` changes Q substantially** — the shipped
Figure-3B trace reads Q = 12.5 at 256 and 79.0 at 512 (where the locked estimator flags the
value resolution-limited and undersegmented rather than reporting it as a linewidth).
`src/spectral.py` defaults to the paper's convention, and `tests/test_paper_values.py` pins it.

The segment length must also suit the *period* of the signal. The 256 default is right for the
simulation traces (periods of 2-6 timesteps) but too short for Figure 1's schematic traces
(120-sample fundamental), where it fits two cycles per segment, fails to resolve the peak, and
collapses Q to 1.0. `make_fig1_concept.py` therefore measures at `nperseg=2048`, which resolves
the fundamental and gives Q = 8.5 — identical across its three panels, which is that figure's
point. Pass `nperseg=` to `analyze()` when your own signal's period calls for it.

## Revision additions (2026)

Files added during peer-review revision:

- `src/rhythmicity_locked.py` — the locked estimator specification behind all revised
  numbers (capped compressive index map, fixed Welch convention, admissibility floors).
- `src/rhythmicity_locked_freq.py` — the same estimator with a fine-grid
  analysis-frequency refinement: the coherence term's frequency is located on a
  zero-padded spectrum rather than the coarse Welch grid, which repairs a quantisation
  failure on long-period signals (an exactly periodic 120-sample signal scores
  c = 1.000 rather than 0.584 at the default grid).
- `src/sweep_design.py` — the complete sweep specification: 5 networks x 5 seed
  locations x 25 α x 20 β x 10 replicates = 125,000 runs, every run's RNG seed a pure
  function of its design position (any run regenerable in isolation).
- `results/per_run_results.csv.gz` — per-run Q (+ validity diagnostics), index
  components and RI for all 125,000 runs.
- `results/per_cell_summary.csv` — per-(network, location, α, β) cell summary with
  Wilson 95% CIs on survival.
- `results/sweep_summary.json`, `results/surrogate_test.json` — headline
  distributional numbers, and the extreme-value test of the maximum Q against
  phase-randomised and AR(1) surrogate families.
- `results/specparam_results.csv.gz` — specparam (Donoghue et al. 2020) fits to every
  surviving run's PSD, joinable 1:1 with the per-run table.
- `data/fig1_panel_values.csv` — measured panel annotations for Figure 1.

## Repository layout

```
rhythmicity-index/
├── src/
│   ├── rhythmicity.py   — the core deliverable: lagged coherence, cycle consistency,
│   │                      compute_rhythmicity_index, classify_rhythmicity
│   ├── spectral.py      — Welch PSD + Q-factor (the baseline RI is compared against)
│   ├── analyze.py       — one-call wrapper: Q and RI for any signal
│   ├── model.py         — Erdős–Rényi graph + Greenberg–Hastings cascade (signal generator)
│   └── eeg_validation.py— EEGMMIDB loader (mne.datasets.eegbci) + per-epoch RI
├── figures/
│   ├── make_fig1_concept.py     — Fig 1: Q is blind to shape (schematic)
│   ├── make_fig2_extinction.py  — Fig 2: extinct–active transition
│   ├── make_fig3_contrast.py    — Fig 3: Q–RI scatter + single-parameter contrast
│   ├── make_fig4_eeg.py         — Fig 4: EEG Berger effect
│   ├── CAPTIONS.md              — run identities and measured values for every panel
│   └── fig*.png                 — rendered figures, 300 dpi
├── data/
│   ├── run_summary.csv          — one row per (network, location, α, β) gridpoint
│   ├── eeg_rhythmicity.csv      — per-subject, per-condition mean RI (20 subjects × {EO, EC})
│   └── example_traces/          — the two Figure-3 traces (.npy and .csv)
└── tests/test_paper_values.py   — regression tests pinning the published numbers
```

### Figure numbering

Figure numbers here match the **published** manuscript. Earlier revisions of this repo used a
different numbering (the extinction figure was "Fig 4", the contrast figure "Fig 2"); those
files have been removed rather than left to confuse. Figure 1 is a schematic and takes no
input data; Figures 2 and 4 read the processed tables; Figure 3 regenerates its second trace
from the model seed.

## Data notes

**`data/run_summary.csv`** is aggregated **per gridpoint** — each row averages the surviving
replicates at one (network, location, α, β) cell. It is the right file for population-level
views, but per-gridpoint means are *not* individual-run values: aggregation flattens extremes.
Individual-run values live in `results/per_run_results.csv.gz` (125,000 rows, the paper's
record); Figure 3 and Table 1 read from it (see `figures/make_fig3_contrast.py`).

**`data/example_traces/fig3B_sharpQ.npy` / `fig3C_highestRI.npy`** are the two Figure-3
traces (run_seed 54202 and 51802). Both are also regenerable in isolation, bitwise-identically,
from their design-position seeds via `src/sweep_sim.py` + `src/sweep_design.py` (net_C =
graph seed 14345, seed node 0; α = 0.421697 / 0.013335, β = 0.05, replicate 2). The previous
revision's exemplar pair (the Q ≈ 63.5 run and the period-3 RI = 3.49 run) is retired: the
former's Q is a resolution ceiling of the spectral grid and lies inside the surrogate-maximum
distribution (`results/surrogate_test.json`), and the latter fails the ≥ 8 samples-per-cycle
admissibility floor.

**Effect size in the EEG validation (revised analysis).** The original submission reported
Cohen's d = 1.08; that number pooled all 600 epochs as if independent (the correct unit of
inference is the subject, n = 20). The revised pipeline (`src/eeg_validation_s4.py`) fixes
this and three further defects: it analyses broadband (1–45 Hz) data with each epoch's
dominant frequency estimated from its own spectrum rather than pre-filtering to 8–13 Hz,
adds condition-blind artifact rejection (blink proxy on Fp1/Fpz/Fp2 + subject-calibrated
amplitude threshold), and computes the analysis frequency with sub-bin refinement rather
than the coarse 0.625 Hz Welch grid. Result (`results/s4_eeg_stats.json`): eyes-open
mean RI 1.178 → eyes-closed 1.904, paired t(14) = 5.49, p = 8e-5, paired dz = 1.42
(15/15 artifact-free subjects increase; without rejection dz = 1.49, 20/20 increase).
The legacy single-filter pipeline (`src/eeg_validation.py`) and its per-subject table
(`data/eeg_rhythmicity.csv`) are retained for provenance; its subject-level dz is 1.69.

## Data & code availability

- **Human EEG:** EEG Motor Movement/Imagery Database (EEGMMIDB), PhysioNet
  (Schalk et al. 2004; Goldberger et al. 2000),
  https://physionet.org/content/eegmmidb/1.0.0/. Downloads automatically via
  `mne.datasets.eegbci`; runs R01 (eyes-open) and R02 (eyes-closed), first 20 subjects,
  occipital channels, 4-second epochs, 8–13 Hz. No EDF files are committed.
- **Simulation:** the processed `run_summary.csv` and one example trace are included. Full raw
  per-run traces (~250k runs) are not committed; `src/model.py` rebuilds any network from its
  seed, so any run is reproducible from (N, K, graph seed, α, β, run seed).

## References

- Cole, S. R., & Voytek, B. (2017). Brain oscillations and the importance of waveform shape.
  *Trends in Cognitive Sciences*, 21(2), 137–149.
- Cole, S. R., & Voytek, B. (2019). Cycle-by-cycle analysis of neural oscillations.
  *Journal of Neurophysiology*, 122(2), 849–861.
- Fransen, A. M. M., van Ede, F., & Maris, E. (2015). Identifying neuronal oscillations using
  rhythmicity. *NeuroImage*, 118, 256–267.
- Greenberg, J. M., & Hastings, S. P. (1978). Spatial patterns for discrete models of diffusion
  in excitable media. *SIAM Journal on Applied Mathematics*, 34(3), 515–523.
- Kinouchi, O., & Copelli, M. (2006). Optimal dynamical range of excitable networks at
  criticality. *Nature Physics*, 2(5), 348–351.
- Schalk, G., McFarland, D. J., Hinterberger, T., Birbaumer, N., & Wolpaw, J. R. (2004).
  BCI2000: A general-purpose brain-computer interface system. *IEEE TBME*, 51(6), 1034–1043.
- Goldberger, A. L., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet.
  *Circulation*, 101(23), e215–e220.

## License

MIT — see [LICENSE](LICENSE).
