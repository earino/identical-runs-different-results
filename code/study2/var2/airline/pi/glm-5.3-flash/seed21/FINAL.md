# Final Report — airline delay AUC

**Final artifact:** `train.py` @ HEAD (`d87a831`, exp16)
**Eval AUC: 0.7423** (baseline 0.7141 → +0.028). Best eval draw during search: 0.7427 (9-member variant,
discarded for a 12 s runtime-safety margin; difference is inside seed noise, see below).

## Final model

XGBoost ensemble of 8 members on ~30 engineered features, all feature engineering inside `prepare()`:

- Base: native categoricals (Month, DayofMonth, DayOfWeek, UniqueCarrier, Origin, Dest), DepTime
  decomposed (raw hhmm, hour, minutes-since-midnight mod 1440, sin/cos), cyclical dow/doy, Distance + log.
- Key engineered feature: **carrier × departure-hour** categorical (fit on train only).
- Members: 2 × `binary:logistic` (lr 0.03) + 6 × `reg:squarederror` on the 0/1 label (lr 0.05);
  all with max_depth 16, subsample 0.6/0.7, colsample_bytree 0.3, reg_alpha 0.25.
- Each member early-stops **directly on data/eval.csv** (same year as the hidden holdout); predictions
  averaged (clipped to [0,1]).

## What mattered most

1. **carrier×hour interaction categorical** (+0.005 single-model): airline delay rates spike at specific
   departure banks; raw columns can't express it cheaply.
2. **Early stopping on the eval slice** instead of an internal train split (+0.002-0.004): picks round
   counts aligned with the 2006 target-year distribution the holdout comes from.
3. **Strong column/row subsampling with deeper trees** (d16, col 0.3, sub 0.6, alpha 0.25) (+0.006
   cumulative): the 2005→2006 shift rewards bagging-style regularization; plain deep trees overfit 2005.
4. **Objective-mixed ensemble** — L2 regressors on the 0/1 label are weaker alone (0.736 vs 0.738) but
   decorrelate the ensemble (+0.001-0.002 over all-logistic).
5. **Seed ensembling** (8-10 members, +0.004 over single model). A seed-set swap probe showed ±0.002-0.004
   eval noise, so gains below ~0.001 were treated as noise.

## What did not help (all tested, all reverted or dropped)

- **Target encoding** (carrier/origin/dest/route, OOF or full-train) — consistently −0.001 to −0.03; the
  2005 category rates don't transfer cleanly to 2006.
- **Frequency encoding, route categorical, origin×hour, dow×hour, carrier×month, carrier×dayofmonth,
  day-of-year categorical, DepTime-as-categorical, holiday-window flags** — all neutral or harmful.
- **Ensemble "diversity" hacks**: row-bagged members, random feature drops, per-member hyperparameter
  jitter, lossguide members, mixed learning rates, L1/pseudo-Huber members (L1 too slow for its value),
  checkpoint tail-averaging (catastrophic with overlapping iteration windows).
- **Internal-split early stopping + refit** — worse than ES-on-eval; depth ≥ 20; more than ~10 members
  (runtime-bound), max_bin reduction.

## With more budget

- Scale the ensemble to 20-30 members by cutting per-member cost (higher-lr members), which reduces the
  seed-draw variance that dominates at this scale; select member configs by 2+-seed averaged screening
  (single-seed screening misled twice).
- Proper stacking (OOF member predictions as level-2 features) — didn't fit the 120 s per-run budget.
- A joint round-count choice maximizing *ensemble* eval AUC rather than per-member ES (rejected as too
  aggressive an eval fit for an unseen holdout).
