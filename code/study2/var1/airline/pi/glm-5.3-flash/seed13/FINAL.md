# Final Report — airline delay (XGBoost, autoresearch harness)

**Best Eval AUC: 0.7590** (baseline 0.7141 → +0.045). Final model: 3-member XGBoost ensemble
(`max_depth=16, learning_rate=0.015, subsample=1.0, colsample_bytree=0.35, es=50`, seeds 42/1/2),
probability-averaged. Validation: `CONTRACT OK`, AUC via `predict_proba` on eval.csv with target removed.

## Changes that mattered most

1. **Deep trees + strong column subsampling + slow learning rate** (0.716 → 0.753 single-model).
   The tuned-shallow regime (depth 6–8, col 1.0) plateaued at 0.716. Sweeping depth × lr × colsample
   revealed a much better regime: depth 14–20, lr 0.01–0.015, colsample_bytree 0.3–0.5. Deeper trees
   only stop overfitting when each split sees ~35–40% of columns.
2. **Time-of-day feature engineering** (~+0.008 with deep models): cyclical sin/cos harmonics (k=1,2,3)
   of minutes-since-midnight, plus `dep_min` and an hour categorical. Departure time is the dominant
   signal; harmonics let shallow-ish splits capture the morning/evening delay peaks.
3. **Carrier × scheduled-hour interaction** (480 categories, +0.005): each airline's delay profile is
   hour-specific. High-cardinality features that encode *year-stable* structure (carrier, hour) helped;
   ones that encode *year-specific* structure (route pair, exact date, target encoding) hurt badly
   because train=2005 vs eval=2006.
4. **Numeric date columns** (+0.007): Month/DayofMonth/DayOfWeek as `c-<n>` strings → integers.
   Numeric DayofMonth/Month generalize across months/years; arbitrary categorical codes do not.
5. **Frequency features + 3-seed bag** (+0.003 combined): log-count of Origin_Dest route, Origin, Dest
   (fit on train only); bagging 3 seeds (±0.002 seed noise) added ~+0.001 and stabilizes the choice.

## Things that did not help

- **Target encoding (OOF, smoothed) of carrier/origin/dest/route**: 0.704 vs 0.714 — 2005→2006 shift
  makes learned per-level rates unstable. Same for route-pair categorical (4198 levels) and a
  Month_Day "date" categorical (0.694).
- **Row subsampling < 1.0 in the final regime** (0.8: −0.0005, 0.9: −0.0004); column subsampling is the
  regularizer that matters here. Also min_child_weight/gamma/lossguide/max_bin: all neutral or worse.
- **Deeper than 16** (d18/d20/d24 ≈ d16 but 20–50% slower) and origin×hour / month×hour / dow×hour
  interactions (all negative). Adding eval.csv rows to training was considered and rejected
  (would corrupt the reported metric and the keep/discard decisions).

## With more budget

- A larger seed bag (5–8 members) at ~25s/member — the per-experiment 120s cap and the CPU budget
  (18,000 CPU-s) were the binding constraints, not wall clock.
- Re-tune the col/lr surface around the final feature set (col 0.35 vs 0.3/0.4 differed by <0.001);
  per-member feature views (e.g., one member without Distance) for cheaper diversity than seeds.
- Possibly pairwise-ranking objectives if a sparse-pair approximation existed; rank- vs probability
  averaging was tested and was a wash.
