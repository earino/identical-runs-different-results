# FINAL — autoresearch XGBoost (airline)

**Best Eval AUC: 0.7458** (validate.sh: CONTRACT OK, AUC 0.7458 reproduced by `predict_proba` on target-stripped eval).

## What mattered most (top changes)

1. **Smoothed target encoding, fit on train only, OOF for the training matrix, maps applied everywhere.**
   The single biggest lever: +0.012 AUC over the best pre-TE model (0.7266 → baseline-TE commit 8689365).
   Smoothed toward the prior with per-key m (e.g. m=100 for Route) so rare keys shrink safely — crucial
   because eval/holdout come from a later year (2006) than train (2005), and unsmoothed TE overfits the
   training year. Self-label leakage was removed by 5-fold OOF encoding of the train rows, then refitting
   the final model on the full training set using the iteration count chosen by early stopping (0.7289).
2. **Time-of-day / hour-keyed TE interactions.** RouteHour + CarrierTOD was the largest single feature
   jump (+0.0084, 0.7341 → 0.7425); DestHour, OriginTOD, DistBin added smaller, real gains (→ 0.7452).
   Delay probability is strongly driven by scheduled departure time interacting with airport/route/carrier.
3. **Multi-seed bagging of the final model** (mean of 8 seeds' predicted probabilities): 0.7451 → 0.7456
   (and 0.7458 with lr 0.025). Cheap, robust variance reduction on top of the TE feature set.
4. **Lower learning rate with early stopping** (lr 0.03 → 0.025, ES on a 15% stratified split of train,
   best_iter≈1756): 0.7456 → 0.7458.
5. **Base feature set from experiment 2** (hour/minute + cyclical sin/cos of minute-of-day, month, day-of-week;
  numeric day-of-month; time-of-day bin categorical; log distance): +0.0023 over the raw baseline.

Final model: mean over 8 XGBoost seeds, depth 8, lr 0.025, subsample/colsample 0.9, lambda 1,
~1750 rounds, on ~40 features (3 native categoricals + DepTimeBin + 13 smoothed TE + RouteCount + numerics).

## What did NOT help (all reverted)

- **Raw interaction categoricals** (Route/CarrierTOD/OriginHour/DestTOD as native categoricals, exp 4/5):
  0.7143 — high-cardinality pairs + depth-8 trees = variance, much worse than TE for the same keys.
- **Date-keyed TE** (Dom, DowHour, MonthDom, OriginDom, CarrierMonth): 0.7189 / 0.7319 — day-of-month and
  month keys do not transfer across the 2005→2006 boundary.
- **Weaker smoothing on Route** (m 100→20): 0.7275 vs 0.7289. Heavier smoothing wins under the year shift.
- Extra TE keys beyond the kept set (RouteTOD, OriginDOW+HourDOW, DestDist/RouteDist, CarrierRoute, HourMin,
  RouteDOW): neutral or worse, and the added columns slowed prepare() enough to risk the 120s timeout.

## With more budget

- Time-decay weighting or per-month TE recalibration to model the 2005→2006 drift directly.
- Quantile-binned DepTime TE keys (finer than hour, coarser than minute) and hierarchical/beta-binomial
  shrinkage (te toward Origin→Route→prior) instead of flat m-smoothing.
- Larger bags (10+ seeds) and depth/lr sweeps jointly with ES budget; OOF-ensemble of TE + non-TE model
  variants as a 2-model blend.
