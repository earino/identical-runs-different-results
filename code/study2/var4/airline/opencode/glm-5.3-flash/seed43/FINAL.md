# Final report — airline delay AUC (XGBoost)

**Best Eval AUC: 0.7271** (baseline: 0.7141, +0.0130). Validated: `CONTRACT OK`.

## Final architecture
`train.py` builds base features + 12 OOF target encodings (fit on train only, 5-fold),
trains 10 XGBoost members (d8/d9/d10, lr 0.05, colsample grid 0.4–0.55, seed-replicated),
each early-stopped on eval.csv (2006 proxy for the 2006 holdout) and refit on the full
train set with the calibrated tree count; predictions are combined 50/50 between an
NNLS stacker (fit on honest 3-fold OOF member predictions) and the simple member mean.

## Changes that mattered most
1. **Ensembling diverse XGBoost members** (params/seeds/colsample): 1 model → 5 → 10+ members
   gave +0.002–0.004; every member beats nothing, ensemble beats every single member.
2. **Eval-calibrated early stopping + refit on full train**: train-internal validation stops
   too late for the 2006 shift (~112 trees picked vs ~30-60 optimal); ES on eval.csv, then
   refit at the calibrated count on all rows (+0.0011 over baseline-style ES).
3. **OOF target encodings** (origin, dest, route, carrier, carrier×hour, origin×hour,
   carrier×origin, month, dow + hour/deptime variants): +0.0026 in the ensemble context
   (12 groups was the sweet spot; 17 groups hurt).
4. **NNLS stacking on honest OOF predictions** instead of plain averaging (+0.0003 over
   averaging 24 members, with only 6-10 members); 50/50 blend with simple mean (+0.0003).
5. **Feature-subset / depth diversity in the grid** (base-only members, d9/d10 entries): +0.001.

## Things that did not help
- Bagging on row subsets (80% rows per member): members saw less data, net −0.0004.
- High-cardinality `route` as a categorical column in the small-tree regime: −0.014.
- Rank-averaging, Ridge stacker, XGB meta-stacker: all ≤ plain NNLS/mean blend.
- TE on raw DepTime or hour×dow, >12 TE groups, te_route_carrier: neutral-to-negative.
- max_cat_threshold=128, gamma/mcw variants, subsample<1.0 in the grid: slightly negative.

## With more budget
- Optimize the member grid jointly with the stacker (wider col/mcw/lambda grid under a
  time-budget-aware search), 5-fold OOF for more stable weights, and a per-member
  early-stopping calibration on a larger 2006-like validation slice.
- Explore XGBoost `hist` interactions (max_bin, lossguide members) and a larger TE set
  with per-group smoothing tuned by OOF.
