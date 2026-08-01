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
but not byte-identical: font availability and matplotlib version affect rasterisation, and two
annotations now show measured rather than hand-carried numbers — Fig 3 prints Q = 63.5 where the
manuscript typeset "≈63" (rounding 63.5 to an integer would print 64 and contradict the text),
and Fig 1's shape-consistency and RI values differ in the second decimal (see the Fig 1 section
below). No underlying data, parameter, or qualitative claim differs anywhere.

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

Two runs over the identical stationary window (**timesteps 500–2000**, well beyond the
transient), each with a 60-step detail (timesteps 1500–1560) so the cycle-to-cycle waveform is
visible. Both panels' annotations are measured at render time by `src/analyze.py` — nothing is
hardcoded, so the figure and Table 1 cannot drift apart.

| Panel | Run | α | β | **Q** | **RI** | class |
|---|---|---|---|---|---|---|
| **3a** | highest-Q run in the entire sweep | 0.75 | 0.10 | **63.5** | **1.62** | WEAKLY_RHYTHMIC |
| **3c** | highest-RI run in the sweep | 0.562341 | 1.00 | **28.3** | **3.49** | RHYTHMIC |

Component measures for 3a: `c` = 0.082, `c̄` = 0.162, `r` = 0.612 — the shape term carries what
little score there is; phase coherence is near zero, which is why a Q of 63.5 does not survive
the AND gate. For 3c: `c` = 0.976, `c̄` = 0.976, `r` = 0.998.

**Provenance.** Panel 3a is the shipped trace `data/example_traces/fig3A_highestQ.npy`. Panel
3c is regenerated from seed: `generate_er_graph(N=20000, K=200, seed=12345)` then
`simulate(alpha=0.562341, beta=1.00, n_steps=2500, init_firing=1, seed=0)`.

**Headline it supports:** ranked by Q, 3a > 3c (63.5 vs 28.3); ranked by RI the order reverses,
3c > 3a (3.49 vs 1.62). The sharpest spectral peak in ~250,000 runs belongs to a signal whose
waveform does not repeat.

**Note on `run_summary.csv`.** These are individual-run values. `run_summary.csv` is aggregated
per gridpoint and its largest Q is 42.5 — the extremes are averaged away — so Table 1 cannot be
read off that table. This is expected, not a discrepancy.

---

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
