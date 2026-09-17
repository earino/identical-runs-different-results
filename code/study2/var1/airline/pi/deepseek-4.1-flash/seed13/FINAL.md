# Final report — airline delay prediction (XGBoost)

**Best Eval AUC: 0.7555** (commit `4f0bf7c`, validated by `./validate.sh` → `CONTRACT OK`).
Baseline was 0.7141, so the search added ~0.041 AUC.

## Changes that mattered most

1. **Time-of-day features from `DepTime`** — extracted `Hour`, `Minute`, `DepTimeNum`,
   `HourSin/Cos`, a 15-minute `RoundDep` slot, plus numeric `MonthNum`, `DayNum`, `DowNum`,
   `DayOfYear`, `IsWeekend`. The raw `hhmm` integer alone under-uses the strongest single signal
   (delays cascade later in the day); this alone lifted the shallow baseline from ~0.7155 to ~0.718.
2. **Frequency / congestion encodings.** Volume counts fitted on train for `UniqueCarrier`,
   `Origin`, `Dest`, `Route`, `Hour`, `Origin×Hour`, `Carrier×Hour`, `Dest×Hour`, and especially
   **`Route×Hour`**. Route-hour volume was the single biggest feature (~+0.004 AUC); the other
   hour-volume features added ~+0.002 more.
3. **Hour-share ratios** (`F_Org_Hour/F_Origin`, `F_Route_Hour/F_Route`, `F_UC_Hour/F_UC`)
   added another ~+0.001.
4. **Deep trees with strong column subsampling.** `max_depth` 18–28 with `colsample_bytree=0.4`
   dramatically beat shallow trees (d5–6 capped near 0.72; d22 reached ~0.745). Deep trees let the
   model combine categorical identity with time/distance/congestion.
5. **`min_child_weight=0/1` + `max_cat_to_onehot=64`.** Lowering the child-weight floor and
   one-hot encoding the low/medium-cardinality categoricals gave a large single-model jump
   (~0.750 → ~0.753). A small **ensemble of deep models (depths 18/22/28, seeds 42–44, half at
   `min_child_weight=0`)** averaging probabilities gave the final ~0.7555.

## Things that did not help

- **`Route` (Origin_Dest) as a categorical feature**: high-cardinality categorical splits
  overfit the 2005 slice and dropped cross-year AUC by ~0.007.
- **Target-encoded delay rates** (OOF) for route/airport/hour levels: 2005 delay propensities do
  not transfer to 2006; every variant was worse (0.70–0.73), confirming that *volume*, not
  *historical propensity*, is what generalizes.
- **Wider/interaction congestion windows**: rolling ±1 h counts and interaction frequencies
  (`Carrier×Origin`, `Carrier×Dest`, `Origin×Dow`, `Route×Month`, …) were all neutral-to-negative
  versus the exact-hour volume features.
- Also unhelpful: `booster=dart` (very slow, untestable in budget), many low-rate trees
  (`lr=0.03, n=400+`) beyond the ensemble, and `max_cat_to_onehot=1000` (all-one-hot).

## What I would try with more budget

The ratio/congestion family looked most promising but hit the CPU cap: the last experiment
(`sh_Dst_Hour` plus route/origin and route/dest share ratios) timed out before reporting, and
those features are cheap and theoretically appealing (they separate "this route is busy" from
"this airport/hour is busy"). I would test that family first, then move to a larger bagged
ensemble (6–10 deep models across seeds, depths and `colsample_bytree`), a time-based inner
validation split to calibrate `n_estimators` honestly, and possibly a logistic stacker over
out-of-fold model predictions. Given the observed year-to-year drift, I would keep avoiding
target encodings and high-cardinality route identity.
