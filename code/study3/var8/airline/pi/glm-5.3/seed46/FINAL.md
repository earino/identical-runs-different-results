# Final report — airline delay classifier (XGBoost)

## Approach

The core challenge is severe time shift: train is 2005, eval/holdout is 2006, so anything that
memorizes 2005-specific structure (route target encodings, route categoricals, calendar
interaction categoricals, early stopping on a 2005 validation split) actively hurts out-of-year
AUC. The winning recipe keeps only physically stable structure: parsed calendar ints,
DepTime decomposed into hour/min/frac numerics, a *hierarchy* of time-of-day categoricals
(hour 24 / half-hour 48 / quarter-hour 96 levels, sharing statistical strength across the
day and smoothing the noisiest dimension), and log-frequency "schedule density" features
(route, origin, dest, origin/dest×hour and ×half-hour, carrier×half-hour). These feed 8
deep (d20–32), strongly regularized trees (eta 0.07, reg_lambda 10, subsample 0.75/0.85,
colsample 0.6–0.8, seeds 1–8) trained for only 125 rounds each, aggregated as a
depth-weighted mean of margins passed through a sigmoid. Diversity across members
(depth × subsample × colsample) was the main post-feature lever; every added diversity
axis helped until it plateaued around 0.756.

## Key experiments

| # | Change | Eval AUC |
|---|--------|----------|
| 1 | Baseline (defaults, X/y as-is) | 0.7141 |
| 2 | Naive: big FE + 2005-internal early stopping (2005 val AUC 0.76) | 0.7109 |
| 3 | Deep regularized trees + hour cat + route/origin/dest freq | 0.7467 |
| 5 | + half-hour-of-day categorical | 0.7525 |
| 9 | Wider ensemble: 8 members (d16–28) @ 125 rounds | 0.7546 |
| 10 | + origin/dest × half-hour density features | 0.7553 |
| 15 | Member subsample 0.75/0.85 (more data per tree) | 0.7559 |
| 23 | Depth frontier: swap d16/d18 members for d30/d32 | 0.7561 |
| 28 | Depth-weighted member aggregation | **0.7562** |

Dead ends (all reverted): route categorical and any target encoding, month×hour /
hour×dow / hour×carrier interaction categoricals, day-of-year, origin/dest×dow density,
more rounds (>150), shallow+many-round, DART/lossguide, sample weights, FE-variant
ensemble members, 9th member (equal, slower).

## Final result

**Eval (2006, 100k) AUC = 0.7562** (baseline 0.7141, +0.042), `validate.sh` → CONTRACT OK,
runtime ~101 s (limit 120 s).

## What I would try with more budget

With ~3 minutes of experiment time slack at the 120 s cap, the runtime is the binding
constraint, so first I would profile and speed up prediction (26 s on 100k rows; e.g.
`inplace_predict`, fewer redundant time cats) to buy room for a 10–12 member ensemble.
Second, I would test time-smoothing regularization — a ±10-minute DepTime-augmented
training copy (kernel-smoothing the time effect at fit time, which unlike test-time
augmentation costs no prediction time) — and a lazy stacking variant that averages
predictions from two independently prepared feature pipelines. Third, I would run the
per-member (depth, subsample, colsample, rounds) mix through a small coordinate-descent
search at ensemble level, since member-composition changes (exps 12/15/23) were the most
reliable micro-lever. Finally, I would quantify eval-noise (±0.001) by re-running the top
three configurations with different seeds to make keep/discard decisions less
overfit-prone.
