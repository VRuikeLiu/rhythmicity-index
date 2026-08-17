# Ground-truth benchmark battery and blinded visual audit

Revision analyses for reviewer items MF-2b (external benchmark with known ground
truth) and REC-15 (systematic blinded visual audit of sweep runs).

- `battery.py` — generators for all 14 signal families (positives: sustained, AM,
  phase-jittered, burst, damped, chirp; negatives: white/pink/brown noise, AR(2),
  spectrum-matched surrogates, shape-randomized; plus one diagnostic arm), the fixed
  parameter grids, and `run_battery()`. Held-out evaluation seeds start at 10,000;
  design-time checks used seeds < 1,000 only. Score each signal with
  `analyze_s2` from `rhythmicity_locked_freq.py` (see `src/`).
- `regen_traces.py` — bit-exact regeneration of the 36 visual-audit traces from the
  sweep's deterministic design-position seeds (`risweep` package layout; adjust the
  import path to `src/` equivalents as needed).
- Results: `../results/benchmark_battery_results.csv` (2,146 signals) and
  `../results/visual_audit_results.csv` (36 runs, blinded ratings).
- Figures: `../figures/fig_benchmark_battery.png`, `../figures/fig_visual_audit_grid.png`.

Blinded rating protocol: each of the 36 traces was rendered identically (no axes,
no metadata, randomized order) and rated 3 times on a 0-10 rhythmicity rubric by a
vision-language model that saw only the image; ratings in the CSV are per-pass
medians. Rating generation requires an LLM API and is not reproducible offline;
the rendered stimuli are regenerable via `regen_traces.py` + the rendering block
documented in the paper's methods.
