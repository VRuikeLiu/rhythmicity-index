# Rhythmicity Index — exact definition

This repository contains the authoritative source code for the **Rhythmicity Index (RI)**
used in the project *"[Neural cascade oscillation study — MSEF]"* by Vincent Ruike Liu,
together with a precise, reproducible statement of the formula.

> **Purpose of this repo.** It is the reference the manuscript's writing assistant asked
> for. The report describes the two component scores *conceptually* but never states how
> they are computed numerically or how they are combined. Everything needed to replace
> that placeholder box is below, and the code that implements it verbatim is in
> [`rhythmicity.py`](rhythmicity.py).

- **`rhythmicity.py`** — the single canonical implementation (**uncapped**; see Version
  note). Contains the two sub-score functions (`compute_lagged_coherence` /
  `_curve`, `compute_cycle_consistency`), `compute_rhythmicity_index`, and the
  `classify_rhythmicity` wrapper.
- **`METHODS_NOTES.md`** — answers to the three manuscript flags: exact dataset identity
  (EEG), threshold usage (RI ≥ 2.0 vs the 1.5 screen), and the 2.2% derivation.

---

## The two component scores

Both are computed on the measurement-window activity time series `A(t)` (post burn-in),
after the Fourier pipeline has estimated the dominant frequency `f_peak` and
`period = fs / f_peak`.

### 1. Phase consistency — *lagged coherence* (Fransen et al. 2015)
> "Do the peaks/troughs recur at the same phase across successive cycles?"

`compute_lagged_coherence(signal, fs, f, n_cycles)`:
1. Split the signal into consecutive non-overlapping windows of length
   `L = round(n_cycles · period)` samples.
2. In each window *k*, mean-subtract and take the single complex Fourier coefficient at
   the frequency of interest:
   `z_k = Σ_t (segment_k[t] − mean) · exp(−2πi · f · t)`.
3. Unit-normalize each coefficient and take the **mean resultant length** (circular
   statistic): `LC = | (1/K) Σ_k z_k / |z_k| |  ∈ [0, 1]`.

Two quantities are derived from this:
- `lc` — lagged coherence at a single **anchor lag of `n_cycles = 3`**.
- `curve_mean` — the mean of the lagged-coherence **curve** over lags
  `n_cycles = 1 … 10` (`compute_lagged_coherence_curve`).

`LC = 0` → phase is random across cycles; `LC = 1` → perfect phase locking.

### 2. Shape consistency — *cycle consistency* (Cole & Voytek 2019, inspired)
> "Does the *waveform shape* of one cycle repeat in the next?"

`compute_cycle_consistency(signal, period)`:
1. **Segment into cycles** by peak/trough alignment: `scipy.signal.find_peaks` with
   `prominence = 0.3 · std(signal)` and `distance = period/2`; use whichever of
   peaks/troughs is more numerous. Keep only inter-extrema segments whose length is
   within `[0.5, 1.5] · period`. (Falls back to rigid equal-`period` bins if fewer than
   2 aligned cycles are found.)
2. **Resample** every cycle to the median cycle length so they are comparable.
3. **Correlate successive cycles**: Pearson `r` between cycle *i* and cycle *i+1*.
4. `mean_corr` = mean of those successive-pair correlations `∈ [−1, 1]`.

`mean_corr` high → the shape genuinely repeats; low → "things going up and down" with no
consistent shape.

---

## How the two are combined into the index

The index is **not** a sum, product, or weighted average. It is a **min-of-two-gates
(logical AND) rule mapped continuously onto three bands**, built so the continuous score
agrees exactly with the discrete RHYTHMIC / WEAKLY_RHYTHMIC / ARRHYTHMIC label.

**Default thresholds** (all configurable; these are the study values):

| symbol | meaning | value |
|---|---|---|
| τ_LC   | strong phase (anchor lag) | **0.5** |
| τ_LCw  | weak phase (anchor lag)   | **0.3** |
| τ_CC   | strong shape              | **0.4** |
| τ_CCw  | weak shape                | **0.2** |
| τ_min  | minimal-evidence floor    | **0.1** |
| (derived) curve-promote / weak-curve | **0.35 / 0.25** |

**Step 1 — normalize each score by its threshold** (so "1.0" means "exactly at
threshold"):

```
strong_phase       = max( lc / τ_LC ,  curve_mean / 0.35 )
weak_phase         = max( lc / τ_LCw,  curve_mean / 0.25 )
strong_consistency = mean_corr / τ_CC
weak_consistency   = mean_corr / τ_CCw
minimal_phase      = max(lc, curve_mean) / τ_min      (0 if ≤ τ_min)
minimal_consist    = mean_corr           / τ_min      (0 if ≤ τ_min)
```

**Step 2 — form the two AND-gates** (a run must satisfy *both* phase *and* shape):

```
strong_gate = min( strong_phase , strong_consistency )
weak_gate   = max( min(weak_phase, minimal_consist) ,
                   min(weak_consistency, minimal_phase) )
```

**Step 3 — map to the banded index** `RI`:

```
if   strong_gate ≥ 1:  RI = 2.0 + (strong_gate − 1)      → RHYTHMIC         (RI ≥ 2.0)
elif weak_gate   ≥ 1:  RI = min(1.0 + (weak_gate − 1), 1.999) → WEAKLY_RHYTHMIC (1.0 ≤ RI < 2.0)
else:                  RI = clip( max(strong_gate, weak_gate), 0, 0.999 )   → ARRHYTHMIC (RI < 1.0)
```

So **RI runs from 0 to ~3**: the integer part *is* the class, and the fractional part
measures how far past (or below) the threshold the weaker of the two gates sits. A run
scores high only when phase consistency **and** shape consistency are *both* strong — a
sharp spectral peak alone is not enough.

### Compact equation

Let `s = min( max(lc/τ_LC, curve_mean/0.35), mean_corr/τ_CC )` (the strong gate) and
`w` the weak gate above. Then

```
      ⎧ 2 + (s − 1)                  if s ≥ 1        (RHYTHMIC)
RI =  ⎨ 1 + (w − 1),  capped at 1.999 if w ≥ 1        (WEAKLY_RHYTHMIC)
      ⎩ max(s, w),    clipped to [0, 0.999)           (ARRHYTHMIC)
```

Classification cutoffs used throughout the paper: **RI ≥ 2.0 = RHYTHMIC**,
**1.0 ≤ RI < 2.0 = WEAKLY_RHYTHMIC**, **RI < 1.0 = ARRHYTHMIC**.

---

## Version note — the cap was removed (canonical = uncapped)

**Decision (confirmed):** the manuscript uses the **uncapped** `rhythmicity.py` above.

An earlier variant capped the RHYTHMIC band at `RI = min(2.0 + (strong_gate − 1), 2.999)`
(which is why an early draft figure showed RI = 2.999). The canonical code removes that
cap: `RI = 2.0 + (strong_gate − 1)`, so the strongest runs can exceed 2.999.

**Removing the cap changes none of the reported numbers.** The cap only affected scores
already ≥ 2.999 — all of which are already RHYTHMIC (RI ≥ 2.0) — so it moves no
classification boundary. Verified against the Stage 4 reclassification data:
**RHYTHMIC = 1,805 / 82,265 survivors = 2.2%** under both variants, and every reported
Spearman correlation and Stage 6 count is unchanged. (A second refinement — the
"continuous minimal-evidence gate," Apr 19 — was likewise validated to preserve the exact
oscillatory-run counts.) No figures need re-rendering, since the affected max-RI traces
are not shown in the manuscript.

---

## References
- Fransen, van Ede & Maris (2015), *Identifying neuronal oscillations using rhythmicity*, NeuroImage.
- Cole & Voytek (2019), *Cycle-by-cycle analysis of neural oscillations*, J. Neurophysiol.
- Donoghue et al. (2020), *Parameterizing neural power spectra (FOOOF)*, Nat. Neurosci.
- Cho et al. (2024), *Cyclic Homogeneous Oscillation detection*, eLife.
