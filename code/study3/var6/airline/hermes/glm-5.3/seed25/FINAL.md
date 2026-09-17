# FINAL — airline dep_delayed_15min (XGBoost)

Best Eval AUC: **0.7626** (experiment #34, commit 0061140; validated: `CONTRACT OK`).
Baseline was 0.7141 → +0.0485.

## The changes that mattered most

1. **Numeric calendar encoding** (exp 8, +0.009): replace the raw `c-<n>` string categoricals for
   Month/DayofMonth/DayOfWeek with plain ints plus sin/cos cyclical terms. The raw c-strings as
   categoricals were actively hurting under 2005→2006 shift.
2. **Traffic-count features** (exps 14–18, cumulative +0.02): unsupervised counts of scheduled
   flights from the training set — per origin/dest/route/carrier at hour, 30-min block, and
   (best) 15-min block granularity, plus log/ratio transforms and distance-vs-route-median.
   Airport schedule congestion is the strongest signal after time-of-day.
3. **Feature-randomized trees** (exps 9–17, cumulative +0.01): colsample_bytree 0.2–0.3 with
   depth 12–14, min_child_weight 10–20, reg_lambda 10. Extreme feature randomness beat every
   "sensible" configuration on the time-shifted eval.
4. **Seed ensembling** (exp 12/34, +0.001–0.003): averaging 3–5 XGB models trained with
   different seeds.
5. **Recency weighting** (exp 22, +0.003): sample weights 1.25^(month-1) so late-2005 months
   count more — the target year (2006) is closer to Dec 2005 than to Jan 2005.
6. **max_bin=1024** (exp 13, +0.001): finer histograms for the dense time features.

## Three things that did NOT help

- **Target/mean encodings** (exp 6 and retry exp 27): OOF target means for carrier/origin/dest/
  route hurt even with heavy shrinkage — the categorical splits already capture this and the TE
  injected 2005-specific rates that shifted in 2006.
- **More capacity via plain boosting**: 400–2000 trees with early stopping on an internal split
  overfit the 2005 data (train AUC 0.84–0.93 vs eval 0.72); DART, hyperparameter-diverse bags,
  colsample_bylevel, incoming-traffic-at-origin features, and 10-min/finer block counts were all
  flat or worse.
- **DOW-conditioned and neighbor-smoothed count variants** (exps 25, 32): diluted the feature
  pool under colsample=0.2 without adding stable signal.

## With more budget

The single biggest remaining lever is a proper rolling-origin validation: all keep/discard
decisions used the single 2006 eval slice, so small (<0.001) calls are noisy. I would build a
month-block CV inside 2005 (train Jan–Oct, validate Nov–Dec, roll forward) to re-rank the last
dozen near-tied experiments. On features: per-airport *day-of-year* traffic profiles (not
day-of-week), scheduled-arrival-time features (DepTime + Distance/500 as an arrival proxy for
the dest airport's congestion), and route×carrier×block count variants are the untested ideas
closest to what worked. On the model: monotone constraints on hour-of-day, and a larger bag
(10–20 seeds at ~350 trees) are the safest increments; a long DART schedule with the recency
weights deserves one more shot at lower learning rate.
