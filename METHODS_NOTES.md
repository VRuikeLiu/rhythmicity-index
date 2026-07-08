# Methods notes — answers to the three manuscript flags

Prepared for the writing assistant. All items below were confirmed directly from the
code and data, not inferred. File/line references are to the `NeuralSimulation` project.

---

## 1. Exact rhythmicity-index formula
See [`README.md`](README.md) and [`rhythmicity.py`](rhythmicity.py). Summary:
- **Phase consistency** = lagged coherence (Fransen et al. 2015): anchor lag `n_cycles=3`
  plus the mean of the coherence curve over lags 1–10.
- **Shape consistency** = cycle consistency (Cole & Voytek 2019 style): Pearson `r`
  between successive peak/trough-aligned, resampled cycles.
- **Combination** = a min-of-two-gates (AND) rule mapped to three bands, **not** a sum or
  product. `RI ∈ [0, ~3)`: `≥2.0` RHYTHMIC, `1.0–2.0` WEAKLY_RHYTHMIC, `<1.0` ARRHYTHMIC.
- Canonical implementation is **uncapped** (see README "Version note").

---

## 2. EEG dataset identity  (FLAG — EEG dataset)

**Dataset:** EEG Motor Movement/Imagery Database (**EEGMMIDB**), PhysioNet — accessed in
code as the MNE "EEGBCI" dataset (`mne.datasets.eegbci`).
Source: `Stage 6/validate_rhythmicity/validate_eeg.py`.

| Item | Value |
|---|---|
| Full dataset | 109 subjects, 64 EEG channels, 160 Hz sampling |
| Subjects used | **20** (S001–S020) |
| Runs used | **R01 = baseline eyes-open (EO)**, **R02 = baseline eyes-closed (EC)** |
| Channels analyzed | occipital **Oz, O1, O2** (averaged) |
| Preprocessing | band-pass 1–45 Hz; 4-second non-overlapping epochs; alpha band 8–13 Hz |
| Local copy | `C:\Users\VRuik\mne_data\MNE-eegbci-data\files\eegmmidb\1.0.0\` (40 `.edf`, S001–S020 × R01/R02) |

**Citations (use both):**
- Schalk, G., McFarland, D.J., Hinterberger, T., Birbaumer, N., & Wolpaw, J.R. (2004).
  *BCI2000: A General-Purpose Brain-Computer Interface (BCI) System.* IEEE TBME 51(6).
- Goldberger, A.L., et al. (2000). *PhysioBank, PhysioToolkit, and PhysioNet.*
  Circulation 101(23):e215–e220.

**Data-availability line (ready to paste):**
> Human EEG data are from the EEG Motor Movement/Imagery Database (EEGMMIDB), PhysioNet
> (Schalk et al., 2004; Goldberger et al., 2000), available at
> https://physionet.org/content/eegmmidb/1.0.0/. Baseline runs R01 (eyes-open) and R02
> (eyes-closed) from the first 20 subjects were analyzed.

---

## 3. Threshold consistency and the "2.2%" figure  (FLAG — thresholds)

Two thresholds are genuinely used, for two different purposes:

| Cutoff | Meaning | Where used |
|---|---|---|
| **RI ≥ 2.0** | "RHYTHMIC" — the primary criterion | Stage 4 main reclassification; EEG validation (`RI_THRESHOLD = 2.0`) |
| **RI ≥ 1.5** | "oscillatory" — a **looser screen** | Stage 6 heterogeneity only (`count_osc(..., threshold=1.5)`) |

The RI ≥ 1.5 screen was needed because at RI ≥ 2.0 the heterogeneity stages produced too
few qualifying runs to correlate (Stage 6 `vary_both`: 79 runs at RI ≥ 1.5; E-I model: 49).

**Recommended wording:** state **RI ≥ 2.0 = "rhythmic"** as the criterion throughout, and
where RI ≥ 1.5 appears (Stages 5–6), describe it explicitly as a *permissive oscillatory
screen* adopted so that enough runs remained for correlation analysis.

### The 2.2% figure — verified from data
The abstract's "only 2.2% … classified as [rhythmic]" is **RI ≥ 2.0**, i.e. the RHYTHMIC
class. Confirmed by aggregating all 322 reclassification chunks
(`Stage 4/outputs/output_robustness_rhythmicity/intermediate/reclass_chunks/`):

| Class (`rhythmicity_class`) | Count | % of 82,265 survivors |
|---|---|---|
| **RHYTHMIC (RI ≥ 2.0)** | **1,805** | **2.2%** |
| WEAKLY_RHYTHMIC (1.0–2.0) | 46,072 | 56.0% |
| ARRHYTHMIC (< 1.0) | 34,388 | 41.8% |

(For contrast, the older spectral pipeline labeled 25,213 survivors "OSCILLATORY" — the
reclassification's point is that spectral peaks alone were far too permissive.)
Primary classification uses the no-FOOOF rhythmicity index; FOOOF was a complementary
check that accounted for < 0.02% of variance, so it does not change these counts.
