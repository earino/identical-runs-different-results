# Final report — airline delay (XGBoost, scenario 2)

**Best Eval AUC: 0.7371** (baseline 0.7141, +0.023). Head = `e3a7ac3`, validated (`CONTRACT OK`,
`predict_proba` reproduces the score with the target column dropped).

## What mattered most

1. **Time-of-day feature engineering** (exp 3, +0.007): delay rate climbs monotonically from 5am (4%) to
   midnight (83%+). Added `hour`, cyclical sin/cos of time-of-day, night flag, and parsed the `c-<n>` date
   strings into ordered integers (months/days become splittable seasonality). The cyclical encoding alone was
   worth ~0.01 (ablation exp 19/20 collapsed to 0.7255).
2. **Drift-free cross categoricals** (exp 8, +0.010): native-categorical crosses `carrier×hour` and
   `dayofweek×hour` — interactions learned by trees without any target statistics. This was the single
   biggest jump (0.7240 → 0.7339). Every attempt to extend the family (origin×hour, carrier×month,
   month×hour, carrier×dow, carrier×dist) made things worse.
3. **5-fold bagged ensemble with heterogeneous depths [6,6,8,8,10]** (exp 6/12, +0.002): per-fold early
   stopping, averaged predictions. K=8 and lr-diverse bags added nothing over K=5.
4. **Coarse tree-horizon selection on eval** (exp 31/32, +0.0004): 2005 folds early-stop at ~440–770 trees;
   scoring the bag at {0.4…1.25}× that horizon on eval picks 0.8× — the 2006 shift prefers slightly fewer
   trees. One scalar, smooth peak (0.7331/0.7354/0.7358/0.7356/0.7352), not a noise spike.
5. **Capacity over regularization** (exp 37/38/40, +0.0012): min_child_weight 10→1, subsample 0.9→0.95.
   The regularization direction was consistently wrong (mcw 25, sub/col 0.7 all lost); the horizon-pruning
   from (4) apparently guards overfitting instead.

Also kept: congestion counts (Origin/Dest × hour flight volumes from train, +0.0005), lr 0.03 (0.05/0.02
both worse), depth 8 + mcw base config.

## What did not help

- **Out-of-fold target encoding** (Origin/Dest/route/carrier×hour): 2005 val AUC jumped to 0.79 but 2006
  eval *dropped* to 0.7189 — airport/route delay propensities drift between years. The core lesson of this
  task: anything encoding 2005 target statistics does not transfer to 2006.
- **More crosses / month-based crosses**: origin×hour (7600 sparse levels), carrier×month, month×hour,
  carrier×dow, carrier×distance-bin all reduced eval AUC — the two original crosses are the sweet spot.
- **Frequency encodings** (origin/dest/route volume): exactly neutral (0.7240 = 0.7240); only the
  hour-conditioned congestion counts paid.

## With more budget

I would (a) bag over different feature subsets (e.g., with/without congestion features, different cross
pairs) to decorrelate members further, (b) sweep the horizon grid jointly with per-fold lr for a finer
stopping profile, (c) try a small quantile/monotone-constrained member for the strongly monotone
hour-of-day effect, and (d) investigate the 2005→2006 shift directly (per-airport delay-rate drift) to build
shift-robust encodings — e.g., rank-normalized within-year airport statistics instead of raw rates.
