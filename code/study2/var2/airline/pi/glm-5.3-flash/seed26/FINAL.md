# FINAL — airline delay AUC (XGBoost, hidden holdout scoring)

**Best Eval AUC: 0.7177** (baseline 0.7141). Final `train.py` = commit `cbe0130`
("caruana: val_size 0.25"); `CONTRACT OK` confirmed via `./validate.sh`.

## Final architecture
Caruana greedy ensemble selection over 16 candidate XGBoost configs (depth 3–7,
lr 0.05–0.12, subsample/colsample 0.7–1.0, native categorical handling), selected
on an internal 75/25 stratified split of `train.csv` (no eval leakage), 40 greedy
rounds with replacement, then all selected members retrained on the full training
set and probability-averaged together with a fixed curated 8-member ensemble.

## Changes that mattered most
1. **Shallower trees (depth 6 → 4)**: 0.7141 → 0.7161. Deeper trees memorize
   2005-specific airport/carrier patterns that don't transfer to 2006 (the eval
   split is time-separated). Depth 3 was slightly worse; 4 is the sweet spot.
2. **Heterogeneous 8-member ensemble** (varied depth/lr/subsample): 0.7161 → 0.7174.
   Diversity across bias regions, not seeds, is what helped.
3. **Caruana greedy ensemble selection** on an internal validation split:
   0.7174 → 0.7175, and with the blend → 0.7176. Pruning/weighting members by
   held-out performance beats uniform averaging.
4. **Blending the selected ensemble with the curated fixed-8 ensemble**: → 0.7176.
5. **Larger internal validation split (20% → 25%)** for more reliable member
   selection: → 0.7177 (30% was worse, 0.7176).

## Things that did NOT help
- **All engineered features**: time-of-day parsing/cyclical encodings, smoothed
  target encoding (m=20–40), route (Origin→Dest) categorical, frequency encodings,
  hour/hour-of-week categoricals, red-eye flag — every single one was neutral or
  worse (route categorical was sharply worse, 0.7094). Raw DepTime as an integer
  plus native categorical handling is already near-optimal here.
- **More capacity**: 100–600 trees, depth 6–8, heavy bagging (sub/col 0.5),
  32-member random-config mega-ensemble — all worse; the 2005→2006 shift punishes
  overfitting immediately.
- **Fancier averaging**: rank-averaging, seed-only ensembles (hist trees are
  near-deterministic without subsampling), DART members, feature-subset
  candidates, two-run selection averaging, min_child_weight/gamma/max_bin=64,
  early stopping with retrain — all neutral or worse.

## With more budget
I would try: (a) a properly time-aware validation scheme — e.g., simulate the
year shift by re-weighting the internal validation set toward rows whose patterns
differ from training (importance-weighted validation); (b) expanding the candidate
pool along the *regularization* axis (per-candidate lambda/alpha/mcw) so the
selector can pick the transfer-optimal regularization, not just the fit-optimal
one; (c) nested selection — run Caruana inside each of several bootstrap resamples
and average the resulting ensembles (bagged ensemble selection); and (d) a
carefully smoothed, CV-fold target encoding evaluated by the selector itself
rather than by hand.
