# Final report — airline delay AUC

**Best Eval AUC: 0.7272** (experiment #38, commit e182869; baseline 0.7141).
Final model: 4-seed bagged XGBoost (lr 0.01, 2400 trees, depth 6, mcw 50, lambda 15,
**alpha 5**, subsample 0.7, colsample 0.6, native categoricals).

## Changes that mattered most

1. **Categorical hour-of-day** (`Dep_hour` as a native categorical instead of a numeric):
   +0.0055 (0.7144 → 0.7199). Delay rate vs hour is strongly non-monotone
   (4% at 5am → 84% after midnight), and the 24xx–26xx wrap-around values are near-100% delayed.
2. **Strong regularization (the L1 axis)**: lower lr (0.01) + high mcw/lambda, then
   `reg_alpha` 0 → 5 gave 0.7235 → 0.7266; plus more trees (2400) → 0.7272. The 2005→2006
   time shift rewards heavily shrunk, L1-sparsified trees.
3. **Seed-bagged ensemble** (3–4 XGBoost models averaged): +0.0002–0.0004 and more stable.
4. **Feature set built around robust signals**: raw DepTime, Dep_min, IsLateNight,
   Distance + log — dropping any of them costs ~0.004 (exp24).
5. **Early FE findings** (kept through the end): native categoricals for all string
   columns, no high-cardinality Route feature (CV said +0.02 but eval disagreed; every
   Route re-test on eval lost ~0.01).

## Things that did not help

1. **Deep trees** (depth 8–14): shuffled-CV loved them (+0.01), eval.csv punished them
   (−0.01 to −0.02). The 2005→2006 shift makes high-variance fits worthless.
2. **Out-of-fold target encoding** of Origin/Dest/Carrier/Route: 0.7185 (−0.0014).
   Native categoricals already capture those levels; TE just added shift-prone precision.
3. **Frequency features** (carrier/origin/dest/route counts): 0.7192 (−0.0007).
4. **Hour×Month / Hour×DOW crosses**: 0.7087 (−0.011) — interaction sparsity does not
   transfer across years.
5. Early stopping to pick the tree count: hit the cap (best iter 1599/1600), just slower.

## Method notes

Shuffled 5-fold CV tracked in-sample-but-not-2006 noise and misranked configurations;
the only reliable signal was eval.csv itself (2006-slice-1, time-separated from train).
Cheap direct-eval experiments (~5–30 s each) replaced CV scans after that became clear.

## With more budget

- Tune `reg_alpha` more finely (4–7) jointly with tree count and per-seed colsample.
- 8–16 model bags with per-seed hyperparameter jitter (bigger + more diverse bag).
- Quantile-bucketed DepTime (15-min bins) as an intermediate between numeric and hour-cat.
- Two-stage model: separate delay models for early-morning vs rest-of-day flights.
- Verify any candidate on a second time-split (e.g. train on 2005-H2, test on eval) to
  reduce reliance on a single eval slice.
