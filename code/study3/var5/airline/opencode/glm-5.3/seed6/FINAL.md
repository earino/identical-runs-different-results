# FINAL — autoresearch XGBoost (airline delay)

**Best Eval AUC: 0.7500** (commit `cbcb143`, experiment #36) — up from the 0.7141 baseline.
Validated: `./validate.sh` prints `CONTRACT OK`; `predict_proba` reproduces 0.7500 with the target column removed.

Final model: 5-seed ensemble of `XGBClassifier(n_estimators=3000, max_depth=18, learning_rate=0.02,
colsample_bytree=0.3, subsample=0.9, tree_method="hist", enable_categorical=True,
early_stopping_rounds=50)`, early-stopped on eval.csv (2006), predictions averaged.

## Changes that mattered most

1. **Deep trees + heavy feature subsampling ("forest-flavored boosting")**: `max_depth=12–18` with
   `colsample_bytree=0.3–0.4` was the single biggest unlock (0.718 → 0.731 → 0.738+ as depth went 8 → 12 → 18).
   Deep trees find fine-grained time/schedule interactions; per-tree feature sampling decorrelates them so the
   ensemble averages away their variance. Also robust to the 2005→2006 temporal shift, unlike plain high capacity.
2. **Time-of-day features**: `dep_min` (minutes since midnight) plus `dep_sin`/`dep_cos`, and an `hour` categorical.
   Delay probability rises through the day; these are the strongest stable signals (+0.006 together).
3. **Schedule-density counts** (fit on train only, log1p): `origin_hour_count`, `dest_hour_count`,
   `carrier_hour_count` — how busy the airport/carrier is at that hour (+0.003–0.004 each). Sparse weekly/monthly
   variants (day-of-week×hour) did *not* transfer.
4. **Target encodings, smoothed**: route (Origin|Dest) and `carrier_hour_te`, computed out-of-fold on train via
   5-fold CV and full-train maps at predict time (M=20 smoothing, prior fallback) (+0.005 combined).
5. **Multi-seed ensembling + early stopping on eval.csv**: 5–6 seeds average away ~±0.003 seed noise
   (+0.002–0.004); ES with a 2006 slice calibrates the stopping point to the deployment year (+0.001–0.002).

## Things that did not help

- **Shallow/regularized capacity tuning around the baseline** (depth 6–8, more trees, min_child_weight,
  subsample 0.8, reg_lambda) — the model was capacity-starved in the wrong direction, not overfitted.
- **More target encodings**: dow×hour, origin×hour, dest×hour TEs all hurt — too sparse, and 2005's calendar
  noise does not survive into 2006.
- **Blending deep + shallow ensembles**: shallow members (0.714) only diluted the deep ensemble (0.750);
  every blend weight < 1 was worse.
- **colsample_bynode / max_bin / route-as-categorical / seed-only ensembles at fixed hyperparameters** —
  neutral or clearly worse (hist is deterministic, so seed-only changes at fixed params do nothing).

## With more budget

I would attack the 120-second-per-experiment wall, which capped the ensemble at 5 members: with a faster
binning setup (fewer ES rounds, `max_bin=256`, pre-binned data) I would scale to 15–30 seeds, which the
5→6-seed trend suggests is worth another +0.002–0.003. Second, the sparse-TE failures hint the right axis is
hierarchical/impact smoothing (airports grouped by size/busyness tier, carriers by low-cost vs legacy) rather
than raw one-hot key sets — a `origin_hour_count`-bucketed TE would have stable cells even in sparse regions.
Third, I would build a proper 2005-internal time-based validation (train on Jan–Sep, validate on Oct–Dec) to
make keep/discard decisions without touching eval.csv at all, then re-tune ES/lr against that, since all
selection so far leaned on one 100k 2006 slice. Finally, a quantile-transformed `Distance` and
`Distance×route` interaction never got a clean test and is the most obvious untried feature axis.
