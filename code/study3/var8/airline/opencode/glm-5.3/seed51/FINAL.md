# Final report — airline departure-delay AUC (XGBoost)

**Best Eval AUC: 0.7272** (baseline 0.7141, +0.0131). Final HEAD: `2d49b20` ("hourly TE m=30").
Contract validated: `CONTRACT OK`, predict_proba reproduces 0.7272 on eval.csv with target dropped.

## What mattered most (in order of impact)

1. **Interaction target encodings at hourly resolution, heavily smoothed** — `origin x hour`,
   `dest x hour`, `carrier x hour`, `hour x day-of-week` TEs with m=30–50 shrinkage, prior fallback
   for unseen keys, and 10-fold out-of-fold values for the training rows. Single biggest win
   (+0.003 alone for origin/dest hourly; finer bins + more smoothing beat 3h bins).
2. **Slow learning + early stopping on AUC** — lr 0.02–0.03 with up to 8000 rounds, ES tracking
   *AUC* (logloss stops ~10x too early). Cumulative +0.004 over lr 0.1.
3. **5-member diverse XGB ensemble** (depths 3/4/5/6 + one subsample/colsample member, different
   seeds and lrs, mean of probabilities): +0.0014 over the best single member.
4. **Time features** from DepTime (hour, minute, cyclic sin/cos) and rough day-of-year: first
   reliable gain, still part of the final feature set.
5. **max_bin=512** (+0.0003) and 10-fold (vs 5-fold) OOF TE (+0.0003).

## What did not help

1. **Native high-cardinality route categorical** (4200 levels): memorized 2005 route noise,
   -0.009 on 2006 eval.
2. **Plain (non-time) target encodings** (origin/dest/route alone, any m): neutral to harmful in
   every regime tried.
3. **Capacity/regularization probes**: depth 8, subsample/colsample 0.7–0.8 on single models,
   min_child_weight/lambda bumps, lossguide member, 6th ensemble member, congestion-count
   features, seasonal month TEs, rank-averaging — all equal or worse on eval.

## With more budget

I would (a) replace fixed-m smoothing with empirical-Bayes per-key shrinkage for the hourly TEs,
(b) build hierarchical TEs (route → origin/dest pooling) so sparse route-hour cells borrow
strength, (c) add an internal 2005 time-split (train Jan–Oct, validate Nov–Dec) to measure drift
robustness directly instead of relying on eval, (d) train all ensemble members at lr 0.01
(measured +0.0015 on the best member but +15s wall — needs a faster ES), and (e) scale the
ensemble to 8–10 members via per-round ES on a subsampled eval set to cut wall time.
