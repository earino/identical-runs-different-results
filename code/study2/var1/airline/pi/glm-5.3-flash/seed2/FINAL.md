# FINAL.md — Airline Delay Classification (dep_delayed_15min)

## Final result

- **Best eval AUC: 0.7266** (final `train.py`, commit `9ee0aac`; validate.sh: **CONTRACT OK**, 0.7266 reproduced via `predict_proba` with the target column removed)
- **Improvement over XGBoost baseline: +0.0125** (0.7141 → 0.7266)
- **Budget used: 40/40 experiments**, ~193 min wall clock, ~6,000/18,000 CPU-seconds.

## Final model

An ensemble of **16 XGBoost `hist` models** (all share `learning_rate=0.05`, `subsample=0.9`,
`max_bin=512`, `tree_method=hist`, `enable_categorical=True`; two slow members use lr=0.03 with
500/600 trees). All members live in one narrow, heavily-regularized deep region:

> depth 11–12, n_estimators 250–350, min_child_weight 50–120,
> colsample_bytree 0.45–0.5, reg_lambda 10–20, reg_alpha 5–10, diverse seeds

Final prediction = weighted average of member probabilities, weights ∝ `(solo eval AUC − 0.72)+`
(quality-weighted; a uniform average ties within noise, 0.7262 vs 0.7266).

Features (all engineered inside `prepare()`, stats fitted on train only):
- raw categorical columns (Month, DayOfMonth, DayOfWeek, UniqueCarrier, Origin, Dest) as pandas
  categoricals with train-fixed levels (unseen levels → NaN);
- `log1p` frequency encodings of carrier / origin / dest / origin→dest route;
- `carorg_freq`: carrier×origin pair frequency (log1p);
- `dep_hour_cat`: hour-of-day from DepTime as a native categorical (raw DepTime kept numeric —
  dropping it costs ~0.018 AUC);
- `log_distance`.

## What the experiments showed (excerpts from experiments.tsv)

| Milestone | Exp | AUC |
|---|---|---|
| Baseline (XGB d6, native cats) | 1 | 0.7141 |
| + frequency encodings | 5 | 0.7150 |
| Target encodings (cross-fit) | 6 | 0.7050 — **worse, abandoned** |
| Shallow (d3) ensemble | 11 | 0.7166 |
| + hour_cat, max_bin 512, carorg_freq | 16 | 0.7172 |
| **Deep + heavy regularization discovered** | 18–22 | 0.7197 → **0.7258** |
| Ensemble scaling 7→9→12→16 members | 26–34 | 0.7256 → 0.7264 |
| Strong-member swap + weighting | 36 | **0.7266** |

## Key learnings

1. **Year-to-year drift dominates.** Train = 2005, eval = 2006, holdout = later 2006. Random
   within-year CV read ~0.76 while cross-year eval sits at ~0.726. Anything that memorizes
   2005-specific rates — target encoding (0.7050), route as a raw categorical (0.7030) — fails
   hard. Models must express *stable* structure (time-of-day, carrier size, route frequency) with
   heavy smoothing.
2. **The single biggest gain** was the deep-regularized region: depth 11–12 with min_child_weight
   50–120, colsample_bytree 0.45–0.5, reg_alpha 5–10 and reg_lambda 10–20 gives +0.009 over any
   shallow/greedy configuration. Deep trees + strong shrinkage + column subsampling effectively
   average out drift-sensitive splits.
3. **Uniform/quality-weighted ensembles of near-equal members** add ~+0.001; adding *weaker*
   members for diversity's sake costs more than the variance reduction gains (exp 24/26).
4. Feature ablations (exp 28–30): every retained feature earns its place; interaction
   categoricals (hour×dow, month×hour, 30-min bins) added exactly nothing (exp 32).
5. OOF stacking overfit (negative meta-coefficients, exp 27); DART timed out (exp 25).
6. Thread-nondeterministic hist fitting gives ±0.0005 AUC run-to-run noise — gains below ~0.001
   are not distinguishable; several "improvements" were ties re-measured.

## Negative results worth recording

- Target encoding of any kind (leave-one-out or cross-fit): catastrophic under drift.
- Route as native categorical: overfits 2005 routes; frequency encoding transfers.
- Interaction categorical features (hour×dow, month×hour, half-hour bins): zero gain over the
  hour categorical plus deep trees.
- Row reweighting/recency tricks: impossible — train.csv rows are shuffled, not chronological.
- By-node/by-level colsample, gamma>0, subsample<0.9: all neutral-to-worse.

## Reproducibility

`train.py` is self-contained, runs in ~95 s (4 threads), prints `Eval AUC: 0.7266`, and exposes
`predict_proba(df)` for the hidden 1M-row holdout. All fit-time statistics live in module-level
maps computed from `train.csv` only; `prepare()` rebuilds every feature for arbitrary DataFrames.
Full log: `experiments.tsv`; per-step code states in git history (40 experiments, 40 commits).
