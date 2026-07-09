# Figure provenance & caption facts

Exact run identities and **canonical** RI values (from `src/rhythmicity.py`, uncapped) for
every panel, so captions are precise. All RI values were recomputed with the canonical
combination; none are the old capped-variant numbers.

## Fig 2 — Q-factor vs. true periodicity (`fig2_contrast.png`)
Three activity traces on a **common 1,000-step window, x-ticks every 250** (direct
comparison). Y-axes differ (different amplitudes) but are labelled identically
("Firing neurons"). No in-plot annotation — all parameters below go in the caption.

| Panel | Network / loc | α | β | replicate | period | Q | **canonical RI** | class |
|---|---|---|---|---|---|---|---|---|
| **2A** highest-Q, not a rhythm | net_B / loc0 | 0.749894 | 0.10 | 1 | ≈2 | **63.5** | **1.24** | WEAKLY_RHYTHMIC |
| **2B** genuine long-period rhythm | net_D / loc1 | 0.005623 | 0.05 | 6 | ≈128 | 0.5 | **2.23** | RHYTHMIC |
| **2C** genuine rhythm | net_E / loc0 | 0.005623 | 0.05 | 3 | ≈69 | 1.1 | **3.14** | RHYTHMIC |

**Panel 2C note:** the trace shown is replicate 3 (RI 3.14, period 69). At this gridpoint
(net_E/loc0, α=0.005623 — sub-critical) only **2 of 10 replicates survive** (rep3 RI 3.14,
rep8 RI 1.999); their **mean RI is 2.57** and mean period 77.5, which are the values in
`run_summary.csv`. The panel renders the shipped trace; it is reproducible from
`src/model.py` via `generate_er_graph(seed=70001)` + `simulate(alpha=0.005623, beta=0.05,
init_firing=[8119], seed=401203)`.

**Headline it supports:** 2A is the highest-Q run in the whole sweep (Q ≈ 63) yet its
canonical RI (1.24) is far below the genuine rhythms (2.23, 3.14) — a sharp spectral peak
does not imply genuine periodicity.

> **Two corrections vs. the earlier draft** (both flow from using the canonical, uncapped
> `rhythmicity.py`, and from tracing the runs to the actual data):
> 1. The report's "α = 0.0056, RI = 2.999" was the **capped-variant** value. The real run at
>    those parameters (net_D/loc1) has **canonical RI = 2.23**. There is no canonical RI of
>    2.999; the maximum canonical RI over the whole sweep is **3.499**.
> 2. These examples live in the **robustness sweep (net_A–net_E)**, not the single "Stage 2
>    network" — captions should name the network/location as above.

## Fig 4 — Extinction / survival threshold (`fig4_extinction.png`)
Survival probability (= 1 − extinction_prob) vs. α on a single frozen network
(**net_A, location 0, β = 0.05, n = 1 fixed initial neuron**); log-x; dashed line marks the
crossing. **α_crit ≈ 0.0075** (first α with survival ≥ 0.5). Consistent with the mean-field
branching-ratio picture (Kinouchi & Copelli 2006: criticality at branching ratio σ = 1;
σ ≈ αK gives α ≈ 1/K = 0.005 for K = 200) — same order of magnitude; do **not** claim an
exact match. Forest-plot / ANOVA / per-network CV panels intentionally dropped.

## Fig 3 — EEG Berger validation (`fig3_eeg.png`)
Per-subject paired mean RI, eyes-open vs eyes-closed. EEGMMIDB (PhysioNet), **20 subjects**,
occipital channels, 4 s epochs. **EO mean RI = 0.566, EC mean RI = 1.523** (Δ = 0.96); the
red line is the condition mean. Matches the reported Berger-effect result.
