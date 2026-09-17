# Final report — airline delay XGBoost (autoresearch scenario 2)

**Best Eval AUC: 0.7240** (baseline 0.7141, +0.0099). Best commit: `5608d8a` ("final mix: 6 A + 2 gentle + 8 deep").
Budget: 40/40 experiments used, ~39 min wall clock left, ~6.2k/18k CPU-seconds used.

## What mattered most

1. **Time-of-day feature engineering** — DepTime decomposed into hour, minutes-since-midnight (with
   correct wrap of 2400–2620 values), and sin/cos of the day angle; plus a small but year-stable
   within-hour 10-minute bucket feature. The delay rate rises almost monotonically from ~0.02 at 5am
   to ~0.8 late evening; this effect alone is worth AUC 0.689.
2. **Robust aux features from train-only statistics**: day-of-year seasonality (sin/cos), log-distance,
   and flight-frequency counts for Origin/Dest/route (hub/traffic proxies — no target involved, so no
   leakage and no train/serve mismatch). ~+0.002.
3. **Bagged XGBoost ensemble** — averaging 16 members with varied seeds, depths (4–10), subsample/
   colsample (0.4–0.8), gamma and learning rate (0.03/0.05), each early-stopped on eval (patience 100).
   Diversity across *hyperparameter families* beat more members within one family. ~+0.004 total.
4. **Deep-tree family with heavy regularization** (depth 9–10, subsample/colsample 0.4, gamma 1) as
   the biggest single late gain: +0.0014 on first contact, +0.0006 per step rebalancing the mix
   toward it (final 6 shallow / 2 gentle / 8 deep).
5. **Removing target encodings**: OOF-smoothed TEs for carrier/origin/dest/route were neutral-to-
   harmful (train rows get noisy fold estimates while new rows get smooth full-map values, and rare
   route cells carry 2005-specific noise). Deleting them was worth +0.0008 and a lot of simplicity.

## What did not help

- **Target encodings in every form tried**: base TEs, interaction TEs (origin/dest/carrier × 3h
  bucket), marginal-rate TEs (hour/month/dow), and a per-hour rate feature with leave-one-out train
  values (that one collapsed to 0.657 — it let trees absorb the hour effect in ~3 rounds and then
  spend all remaining capacity overfitting fragile residual signal).
- **GAM-style base_margin offset** (hour-rate logit as starting margin via xgb.train): −0.005.
- **Hyperparameter churn within one family**: lr 0.03 vs 0.05, min_child_weight 10, reg_lambda 2,
  max_bin 512, 24 members, logit-vs-prob averaging — all statistically indistinguishable (0.7205
  across ~7 configs). Monotone constraint on a "minutes since 5am" ramp feature: −0.0015.
  Lossguide members and forest-flavored (num_parallel_tree) members: slightly worse.

## With more budget

I would explore the deep-regularized family much harder — it was still yielding +0.0006 per rebalance
when the budget ran out (depth 11–12, subsample 0.3, gamma tuning, more members, and a fourth family
trained on feature-disjoint subsets). Second, I would build a proper 2005-internal validation design
(e.g., month-held-out CV) to sanity-check that eval-based early stopping isn't subtly selecting
rounds that flatter 2006-slice-1, since the hidden holdout is a different 1M-row slice. Third, I would
revisit *unsupervised* route/airport embeddings (count statistics, co-flight schedule density) — the
leakage-free statistics were the only features that consistently generalized across the year shift.
The eval standard error (~±0.0016 at 100k rows) exceeded most single-change effects, so I would also
run 2–3 seeds per candidate before keep/discard decisions.
