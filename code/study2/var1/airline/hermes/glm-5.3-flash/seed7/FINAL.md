# FINAL — autoresearch XGBoost (airline, scenario 2)

**Best Eval AUC: 0.7348** (experiments.tsv #35, commit `a937d39` — HEAD).

Final architecture: `prepare()` inside the contract path builds raw categoricals (Month, DayofMonth,
DayOfWeek, UniqueCarrier, Origin, Dest), DepTime parsed as hhmm (leading-zero loss handled via
`%1440`) into dep_hour + cyclical sin/cos, log1p(Distance), and interaction categoricals
(route, carrier×hour, origin×hour, dest×hour, dow×hour, 2h-block origin/dest×hour, dom bucket).
Model: XGBClassifier hist/enable_categorical, depth 24, lr 0.02, subsample 0.4, colsample_bytree 0.2,
reg_lambda 10, early stopping on a random 80/20 split of train (best_iter≈600), refit on all data with
3 seeds averaged in `predict_proba` (explicitly allowed XGBoost-only ensemble).

## Changes that mattered most

1. **Stochastic regularization sweep (subsample/colsample)** — the single biggest lever: 0.7241 →
   0.7299 as subsample 0.8→0.4 and colsample 0.6→0.2. With year-drift (2005 train → 2006 eval),
   preventing any single feature (esp. high-cardinality Origin/Dest) from dominating splits
   generalizes far better.
2. **Capacity + real early stopping** (#4, #9–#10, #29–#33): ES on a held-out 20% of train with
   refit at best_iter, then depth 6→24, lr 0.1→0.02. Depth 24/lr 0.02 ≈ 0.7342 (+0.010 over depth 8).
3. **Interaction categoricals** (#6, #7, #14): route, carrier×hour, origin/dest×hour, dow×hour,
   dom bucket → 0.7143→0.7241 cumulative. Lets trees learn airport/time context directly.
4. **reg_lambda 2→10** (#25): +0.0016 on top of the stochastic sweep.
5. **3-seed bag of the refit model** (#35): +0.0006 (0.7342→0.7348); averages out the
   high-variance column-sampling draws at colsample 0.2.

## What did not help

- **Out-of-fold target encoding** of carrier/origin/dest/route (#5, 0.7117 vs 0.7167): target
  statistics from 2005 do not transfer to 2006 (origin-rate year-correlation only 0.38).
- **Coarser time blocks / calendar features**: 2h-block × origin/dest (#18), month×hour +
  dep_period (#15, retried #36 under heavy reg), days-to-holiday (#16) — all slightly negative.
- **grow_policy="lossguide"** (max_leaves 128, #17): 0.7201 vs 0.7203 depth-wise at equal budget.
- **Overshoots of the optima**: depth 28, λ 20/30, mcw 10/20, subsample 0.3, lr 0.01 (timeout) —
  all worse; 5-fold-median best_iter (#12) and ES-on-AUC (#13) were within noise of the simple
  80/20 logloss ES but 3–5× more compute, so discarded per the simplicity rule.

## With more budget

- 5×2-seed bag with per-member bootstrap-resampled train rows, and/or bagging over the ES split
  (each member ES on a different fold, refit on the rest) — variance reduction was still paying.
- Categorical join of Origin×Dest×hour-block (3-way) and Origin×carrier, tested under the heavy
  regularization regime (the light-reg regime punished new categorical features; that verdict may
  not hold anymore, as the month×hour retry showed).
- Per-origin ridge-adjusted DepTime cluster features, and a monotone constraint on
  days-to-holiday-style recency features; lr 0.015 with a 240s allowance.
- A small 2D grid around (subsample, colsample) = (0.4, 0.2) × λ ∈ {8, 10, 12} with the bag,
  since single-fit measurements there had ±0.001 noise.
