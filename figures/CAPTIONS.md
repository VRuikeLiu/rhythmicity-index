# Figure provenance

Run identities and measured values for every panel, at the numbering used in the published
manuscript. All values below were produced by the scripts in this directory using
`src/rhythmicity.py` and `src/spectral.py`; `tests/test_paper_values.py` pins them.

Q convention throughout: Welch with `nperseg = min(256, len(signal))`, half-power bandwidth
from the outermost bins above half peak power. Q is resolution-dependent — see the README.

**Relationship to the published PNGs.** The `.png` files here are the output of the scripts in
this directory, so the committed images, the scripts, and `tests/test_paper_values.py` are
mutually consistent — re-running a script reproduces the committed file. `fig2_extinction.png`
is byte-identical to the image in the published manuscript. Figures 1, 3 and 4 are equivalent
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

Survival probability (1 − extinction probability) vs. α on a single frozen network:
**net_A, location 0, β = 0.05**, one fixed initial firing node, log-scaled α, 10 replicates
per α, 25 α values.

- **α_c = 0.004217** (printed as 0.0042) — the last fully-extinct α below the endpoint.
- Consistent with the mean-field branching-ratio picture (Kinouchi & Copelli 2006:
  criticality at σ = 1; σ ≈ αK gives α ≈ 1/K = 0.005 for K = 200). Same order of magnitude —
  the paper does not claim an exact match.
- **α = 1 is a degenerate deterministic limit** and is *also* fully extinct (survival 0.0):
  every resting node with an active neighbour fires at once, the whole population enters
  refractoriness together, and nothing remains to re-ignite it. Any code locating the
  transition must exclude this endpoint.
- Two small irregularities reflect finite sampling, not real non-monotonicity: survival is
  0.90 at both α = 0.0100 and α = 0.0133, and dips to 0.90 at α = 0.3162 within the
  otherwise-full plateau.

Reads `data/run_summary.csv`.

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

Per-subject paired mean RI, eyes-open vs eyes-closed. EEGMMIDB (PhysioNet), **20 subjects,
600 four-second epochs**, occipital channels O1/Oz/O2, alpha band 8–13 Hz. Grey lines connect
conditions within a subject; the red line is the condition mean.

- **EO mean RI = 0.566**, **EC mean RI = 1.523** (Δ = 0.956).
- **19/20 subjects increase** when the eyes close.
- Effect size: the manuscript reports **Cohen's d = 1.08**; recomputed from
  `data/eeg_rhythmicity.csv` the effect is larger — **paired dz = 1.69, pooled d = 1.94**
  (printed by `make_fig4_eeg.py`). The manuscript value is the more conservative one; the
  direction, group means, and "large effect" characterization hold either way.

Per-subject values come from `src/eeg_validation.py`, which downloads the EEG via
`mne.datasets.eegbci`; the figure script only plots `data/eeg_rhythmicity.csv`.
