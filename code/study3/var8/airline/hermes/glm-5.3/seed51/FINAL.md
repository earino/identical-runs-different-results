# Final report — airline delay XGBoost (autoresearch benchmark)

## Result

- **Best Eval AUC: 0.7574** (experiment #12, commit `a8dd8cb`, validated: `CONTRACT OK`).
- Baseline (commit `57948ad`): 0.7141 → **+0.0433** improvement.
- Budget at stop: 13/40 experiments used, ~127 wall-clock minutes remaining, but **CPU compute was
  the binding constraint**: 17,652 of 18,000 CPU-seconds spent. The remaining ~350 CPU-s cannot
  cover another experiment plus the mandatory final validation, so the run was finalized with
  budget left, as program.md requires ("When it is spent, Python refuses to start, so finalize
  before that").

## The 3-5 changes that mattered most

1. **Deep, low-rate trees** (`max_depth=20`, `learning_rate=0.02`, `min_child_weight=1`,
   `colsample_bytree=0.6/0.5`, `subsample=0.8`, `reg_alpha=1.0`): the single biggest jump
   (0.715 → ~0.74 offline; combined with the features below it carried the run). Shallow/lossguide
   trees and higher learning rates were all worse; the AUC kept improving as depth went 8→12→14→20.
2. **Departure-time bucket categoricals**: hour-of-day as a 25-level categorical and a 97-level
   15-minute-bucket categorical (on top of numeric `hour`, `mins`, sin/cos, minute-of-hour).
   +0.003 each; 15-min buckets were the sweet spot — 5-min buckets overfit (0.7467).
3. **Dropping month/day-of-month**: 2005 calendar effects do not transfer to 2006; removing them
   gave +0.001 and a simpler feature set. Day-of-week was kept (transferable).
4. **3-member XGBoost ensemble** (fixed 400/400/320 rounds, diverse colsample/subsample/seed,
   averaged probabilities): +0.0005-0.001 over a single model, and it replaced the fragile
   early-stopping+refit pattern, cutting wall time variance near the 120s cap.
5. **Plain categoricals over target encodings**: leaving UniqueCarrier/Origin/Dest as native
   XGBoost categoricals beat every smoothed target-encoding variant I tried (route TE, origin-hour
   TE, count features). The model extracts the signal itself; pre-summarized encodings lose
   interaction capacity.

## What did NOT help (all verified, then reverted)

- **Target encodings** (carrier/origin/dest/route/hour, alpha 20..500, with/without OOF):
  0.7087-0.7140 vs 0.7170 plain cats. Consistently worse on the time-shifted split.
- **Route as a categorical / origin-hour and dest-hour pair categoricals / route distance stats**:
  high-cardinality combinations fragmented the data (0.7044-0.7522 — all below baseline config).
- **Lossguide growth policy, max_bin=512, month sin/cos, day-of-year, date-as-categoricals**:
  all neutral or worse.
- **reg_alpha=2.0** (last experiment): 0.7568 < 0.7574 — the optimum is near 1.0.

## With more budget I would try

More CPU-seconds, not more experiments, was the wall: each in-repo experiment cost ~450-800
CPU-s (mostly the 3-member ensemble fit), and offline probing cost the same compute. With more
budget I would (a) tune the ensemble members jointly — different feature subsets per member
(bagged features) rather than only colsample diversity; (b) test 2005-month-of-year → 2006
recency weighting or sample weights, since the train/test shift is temporal; (c) sweep
`max_cat_threshold`/`cat_smooth`-style options XGBoost 3.4 exposes for optimal-partition
splits; (d) probe a wider ensemble (6-10 cheaper members, depth 14-16) where the per-member
compute fits, since averaging kept adding small gains; (e) revisit distance features — raw and
log-distance were both kept, but no interaction (e.g. distance × hour, distance × carrier)
was tested.
