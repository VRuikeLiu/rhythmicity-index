# Figure provenance

Run identities and measured values for every panel, at the numbering used in the published
manuscript. All values below were produced by the scripts in this directory using
`src/rhythmicity.py` and `src/spectral.py`; `tests/test_paper_values.py` pins them.

Q convention throughout: Welch with `nperseg = min(256, len(signal))`, half-power bandwidth
from the outermost bins above half peak power. Q is resolution-dependent — see the README.

**Relationship to the published PNGs.** The `.png` files here are the output of the scripts in
this directory, so the committed images, the scripts, and `tests/test_paper_values.py` are
mutually consistent — re-running a script reproduces the committed file. (An earlier note that
`fig2_extinction.png` was byte-identical to the as-submitted manuscript image applied to the
original single-network figure; the revision replaced that figure — see the Fig 2 section.)
Figures 1, 3 and 4 are equivalent
but not byte-identical: font availability and matplotlib version affect rasterisation, and
annotations are measured rather than hand-carried — Fig 3's panel values are asserted against
the sweep delivery table at render time, and Fig 1's shape-consistency and RI values differ in
the second decimal (see the Fig 1 section below). No underlying data, parameter, or qualitative claim differs anywhere.

**Measured vs. hardcoded.** Every annotated number in every figure is now computed at render
time from the plotted trace. Nothing is hardcoded — which is what let the figures and Table 1
drift apart in earlier revisions of this repo.

---

## Fig 1 — Q is blind to waveform shape (`fig1_concept.png`)

**Schematic traces, but measured values.** The three traces are synthetic (not simulation
output), constructed to share identical regular timing — a 120-sample fundamental in every
panel — so their spectral peaks, and therefore their Q-factors, come out the same. Only shape
consistency differs. All annotated values are measured at render time by `src/analyze.py`.

| Panel | Construction | phase coherence | shape consistency | Q | RI | class |
|---|---|---|---|---|---|---|
| **1a** shape varies each cycle | harmonics re-randomised every cycle (`mix = 0.00`) | 1.00 | 0.38 | **8.50** | **2.00** | WEAKLY_RHYTHMIC |
| **1b** shape mostly repeats | interpolated (`mix = 0.55`) | 1.00 | 0.74 | **8.50** | **2.84** | RHYTHMIC |
| **1c** shape repeats exactly | one fixed harmonic set (`mix = 1.00`) | 1.00 | 0.97 | **8.50** | **3.42** | RHYTHMIC |

Q is **identical to two decimals across all three panels (8.50)** — which is the figure's whole
point — while RI rises monotonically with shape consistency and crosses the rhythmic boundary
between 1a and 1b.

**Spectral resolution for this figure.** Q must be measured at `nperseg = 2048` here, not the
paper's default 256. The default suits the simulation traces, whose periods are 2-6 timesteps;
against a 120-sample fundamental it fits only two cycles per segment, fails to resolve the peak,
and returns Q = 1.0 for all three panels. At 2048 (~17 cycles per segment) the fundamental
resolves and Q = 8.5. This is the same resolution-dependence documented in the README, and a
concrete illustration of why the segment length must suit the period of the signal in hand.

**Small differences from the published caption.** The manuscript caption reports shape
consistency 0.35 / 0.76 / 1.00 and RI 2.00 / 2.90 / 3.50; measurement gives 0.38 / 0.74 / 0.97
and 2.00 / 2.84 / 3.42. The differences are in the second decimal, arise because the published
caption values were carried by hand rather than read from a measurement pass, and change
nothing about the figure: Q identical across panels, phase coherence 1.00 throughout, RI rising
monotonically, and 1a below the rhythmic boundary with 1b and 1c above it. The repo reports the
measured values because they are what the committed code produces.

**Headline it supports:** Q responds only to timing regularity, so it cannot distinguish the
non-repeating signal in 1a from the true rhythm in 1c. RI, which additionally requires shape
consistency, separates them.

---

## Fig 2 — Extinct–active transition (`fig2_extinction.png`)

**Revision (S6).** Two panels, computed from the locked-spec sweep's per-cell table:
every survival point pools **250 runs = 5 network realisations × 5 seed-node locations ×
10 replicates** (the original figure used 10 replicates on one frozen network with one
seed node). **A)** survival vs. α at five β values (0.05, 0.25, 0.50, 0.75, 1.00) with
95% Wilson CIs and the mean-field 1/K reference line. **B)** critical point fitted
separately at each of the 20 β values (95% profile-likelihood CIs).

- **α_C = (5.03 ± 0.03) × 10⁻³** (95% CI) — maximum-likelihood fit of the mean-field
  branching-process survival probability (s = 1 − exp(−(α/α_C)s)) to the 8 grid points
  spanning the transition, pooled over β. Within 0.6% of the Kinouchi & Copelli (2006)
  prediction α ≈ 1/K = 0.005 for K = 200.
- The previously quoted **0.0042** is the last fully-extinct grid value (0 of 30,000 runs
  survive at α ≤ 4.2 × 10⁻³): a *lower bound* whose 16% distance from 1/K is set by the
  0.125-decade grid spacing, not a critical-point estimate.
- **β-independence** (previously asserted, now shown): the five survival curves coincide;
  per-β fitted α_C has no trend (Spearman ρ = 0.24, p = 0.30; max deviation from 1/K
  2.7%); survival-vs-β rank correlation over all 500 grid cells ρ = 0.01 (p = 0.82).
- **α = 1 is a degenerate deterministic limit** and is *also* fully extinct (all 5,000
  runs): every resting node with an active neighbour fires at once, the whole population
  enters refractoriness together, and nothing remains to re-ignite it. Any code locating
  the transition must exclude this endpoint.
- The old caption's "two small irregularities" were n = 10 sampling noise and are gone at
  n = 250. One real feature remains: survival dips to **0.940 (Wilson CI 0.903–0.963)** at
  α = 0.32, β = 0.05 (15/250 extinct, spread over all 5 networks and 5 locations; plus
  2/250 at α = 0.42, β = 0.05; zero extinctions in 0.042 ≤ α ≤ 0.75 at any β ≥ 0.10).
  Initial-cascade fizzle is impossible there (branching ratio αK ≈ 63), so these runs die
  by later near-synchronous collective collapse into refractoriness — the α = 1 mechanism,
  reachable stochastically at the lowest recovery probability.

Reads `results/per_cell_summary.csv`; writes `results/fig2_alphaC_fits.csv` alongside the
PNG. The old single-network figure and its `data/run_summary.csv` input remain in the repo
history.

---

## Fig 3 — Spectral sharpness vs. genuine repetition (`fig3_contrast.png`) — **and Table 1**

Three panels. **A)** the joint distribution of Q and RI over the 11,938 admissible surviving
runs (Spearman ρ = −0.005): spectral sharpness carries no rank information about waveform
repetition, and every top-percentile-Q run falls below the rhythmic threshold. **B)** and
**C)** a single-parameter contrast: identical design (net_C, seed node 0, β = 0.05,
replicate 2) except α. Panel annotations quote the sweep delivery table (`results/`), the
paper's record; the script re-measures both traces at render time and asserts Q, c and the
period agree with the table.

| Panel | Run | α | β | **Q** | **RI** | class |
|---|---|---|---|---|---|---|
| **B** | sharp-spectrum comparison (run_seed 54202) | 0.421697 | 0.05 | **12.491** (94th pctile) | **1.342** | WEAKLY_RHYTHMIC |
| **C** | highest-RI admissible run (run_seed 51802) | 0.013335 | 0.05 | **2.398** | **2.378** | RHYTHMIC |

Component measures for B: `c` = 0.229, `c̄` = 0.190, `r` = 0.304 — a sharp, honestly-resolved
peak (6.8 bins across) whose phase and shape consistency all sit below their thresholds. For
C: `c` = 0.568, `c̄` = 0.545, `r` = 0.551 at a measured period of 26.08 timesteps (26.1
samples/cycle, 57.5 cycles in the window).

**Provenance.** Both traces ship in `data/example_traces/` and regenerate bitwise-identically
from their design-position seeds (`src/sweep_sim.py`, `src/sweep_design.py`).

**Headline it supports:** ranked by Q, B > C (12.49 vs 2.40); ranked by RI the order
reverses, C > B (2.378 vs 1.342) — with everything but α held fixed. The distributional
panel shows this is the rule, not an anecdote: ρ(Q, RI) = −0.005 across the admissible sweep.

**Retired exemplars.** Earlier revisions contrasted the sweep-maximum-Q run (Q ≈ 63.5,
α = 0.75, β = 0.10) with the period-3 highest-RI run (RI = 3.49, α = 0.562, β = 1.00). Both
are retired: the former's Q is resolution-limited (a ceiling of the spectral grid — measured
on an adequately resolved window the same run gives Q ≈ 1.0) and lies inside the
surrogate-maximum distribution (`results/surrogate_test.json`); the latter fails the ≥ 8
samples-per-cycle admissibility floor (a 3-sample cycle has zero shape degrees of freedom).

**Note on `run_summary.csv`.** These are individual-run values. `run_summary.csv` is aggregated
per gridpoint and its extremes are averaged away, so Table 1 cannot be read off that table.
This is expected, not a discrepancy.

## Fig 4 — EEG Berger validation (`fig4_eeg.png`)

Per-subject mean RI, eyes-open vs eyes-closed, from the REVISED broadband pipeline
(`src/eeg_validation_s4.py` + `src/eeg_score_all_s4.py`). EEGMMIDB (PhysioNet),
**20 subjects, 600 four-second epochs total across conditions**; occipital channels
O1/Oz/O2 averaged after 1–45 Hz zero-phase FIR filtering; per-epoch dominant frequency
estimated freely over 3–20 Hz (not fixed to the alpha band); condition-blind artifact
rejection (blink proxy on Fp1/Fpz/Fp2, subject-calibrated occipital amplitude
threshold) leaves 415 accepted epochs of which 365 pass the admissibility floors.

- Filled circles: 15 subjects with artifact-free data in both conditions (grey lines
  connect within subject). Open circles: eyes-closed means of the 5 subjects with no
  artifact-free eyes-open epochs (excluded from paired tests).
- Red squares: condition means ± 95% CI over the paired n = 15.
- **EO mean RI = 1.178**, **EC mean RI = 1.904**; Δ = 0.726 (95% CI 0.442–1.010);
  paired t(14) = 5.49, p = 8e-5; Wilcoxon W = 0, p = 6e-5; **paired dz = 1.42**,
  pooled d = 1.63; **15/15 paired subjects increase**. Without rejection: dz = 1.49,
  20/20 increase.
- Dashed reference lines at RI = 1 (weakly rhythmic) and RI = 2 (rhythmic).
- Numbers: `results/s4_eeg_stats.json`; per-epoch table `results/s4_eeg_per_epoch.csv.gz`;
  per-subject table `results/s4_eeg_per_subject.csv`.

The legacy narrowband pipeline (`src/eeg_validation.py`, `data/eeg_rhythmicity.csv`,
plotted by `make_fig4_eeg.py`) is retained for provenance of the originally submitted
figure; the original submission's d = 1.08 was epoch-level pooling of that pipeline's
600 values.
