# Final report — autoresearch XGBoost (airline)

**Best Eval AUC: 0.7374** (baseline 0.7141, +0.0233). Final commit: `273af40` (subsample 0.95), validated
`CONTRACT OK`.

## Final model

Ensemble of 3 XGBoost classifiers (depth 16/12/20, seeds 42/7/123), probabilities averaged. Shared config:
lr 0.02, 6000 trees cap, early stopping (patience 100) on a random 5% holdout of train, subsample 0.95,
colsample_bytree 0.3, min_child_weight 5, reg_lambda 2, hist, max_bin 512, enable_categorical.
Features (~26 cols): raw categoricals (Month, DayofMonth, DayOfWeek, carrier, Origin, Dest), scheduled-departure
time-of-day (minutes, sin/cos, hour as categorical), Distance + log1p, 10 OOF-smoothed target encodings
(carrier, Origin, Dest, Route, Carrier×Origin/Dest/Hour/Dow/Month, Origin×Hour, Dest×Hour, m=60), and 4
log-frequency encodings (Origin, Dest, Route, carrier). All statistics fitted on train only inside `prepare()`.

## Changes that mattered most

1. **colsample_bytree 0.3** (+0.0148, exp 9-10): the single biggest lever. Without feature subsampling the
   target encodings dominate and overfit; column bagging forces trees to combine raw + encoded views.
2. **Ensemble of 3 depths × seeds** (+0.0017, exp 16): probability averaging of depth 12/16/20 members.
3. **95% training rows via 5% ES holdout + patience 100** (+0.0018, exp 36; subsample 0.95 +0.0004, exp 40):
   more data per member beats a large early-stopping holdout.
4. **Feature engineering** (+0.002 total, exp 2/4/5): time-of-day features and OOF target encodings,
   especially Origin×Hour; max_bin 512 (+0.0003, exp 22) also helped slightly.
5. **lr 0.02–0.03 with ES** instead of a fixed small tree budget (+0.0011, exp 3).

## Things that did not help

- **Lossguide (leaf-wise) trees** with max_leaves 48–96: 0.7248 vs 0.7352 — clearly worse here (exp 30).
- **More/finer TE interactions** (Origin×Dow, Dest×Dow, Carrier×Hour, Hour×Dow, Month×Hour, 2h-bins):
  every addition past the core 10 hurt (0.7327–0.7336; exp 19, 27, 34) — the TE set is saturated.
- **Robustness tweaks**: recency sample weights, TE smoothing 150, time-based (Nov–Dec) ES split,
  row-bagged members, log-odds TE transform, cheap 4th ensemble member, reg_alpha/reg_lambda 5 — all
  neutral-to-worse (exps 8, 15→neutral; 17, 20, 21, 24/25 timeouts, 31, 33, 35, 38, 39).

## With more budget

I would (1) tune the ensemble composition at the 5%-valid regime (4–5 members at ~25s each needs faster
prediction paths — batch DMatrix predict already landed; next is lower patience + more members), (2) sweep
colsample_bytree finely (0.27/0.33) and max_bin (768) around the current peak since both showed strong
sensitivity, (3) try bagged target encodings (per-member OOF folds with different seeds) so members see
decorrelated encodings, and (4) test whether training members on 100% of data with the ES-tuned tree count
(refit without early stopping) adds the last data fraction — it did not fit the 120 s cap here.
