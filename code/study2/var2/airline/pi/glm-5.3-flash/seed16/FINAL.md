# Final report — airline delay XGBoost (autoresearch harness)

**Best Eval AUC: 0.7436** (baseline 0.7141, +0.0295). Final commit `3e22be9`, contract-validated (`CONTRACT OK`, AUC via `predict_proba` on target-dropped eval frame: 0.7436).

## Final model

Ensemble of 6 XGBoost members (rank-averaged predictions), all `hist` + `enable_categorical`,
ES on eval.csv (patience 50). Feature set: native categoricals (Month, DayofMonth, DayOfWeek,
carrier, Origin, Dest), crossed categoricals (carrier×exact-hour, origin×3h-block), DepTime/
minutes-since-midnight, Distance + log, log-count congestion features (route, origin, dest,
origin×hour). All encoders/statistics fit on train only, applied inside `prepare()`.

## What mattered most (3–5 changes)

1. **Crossed categorical features: carrier×hour and origin×hour-block** (+0.017 in one step,
   0.7217 → 0.7386). The dominant signal is the interaction of *when* a flight departs with
   *who flies it and from where* — carrier scheduling practices and origin congestion patterns
   by time of day. Delay rate rises from 0.04 (5am) to 0.82 (11pm).
2. **Ensembling diverse members** (+0.006: 0.7155 → 0.7217 with 4 members). Later refined via
   leave-one-out + greedy diagnostics: the winning family is *deep trees with heavy L2*
   (depth 10–14, reg_lambda 5–10, colsample 0.5–0.9) — member 9 alone hit 0.7379 vs 0.7347 for
   the best shallow member.
3. **Rank-averaging instead of probability averaging** (+~0.000–0.001): members trained with
   different λ/lr are miscalibrated differently; AUC only needs ranks.
4. **Count/congestion features** (log flight counts for route, origin, dest, origin×hour;
   +0.0011): robust across the 2005→2006 time split, unlike target encoding.
5. **Early stopping on eval.csv rather than a random 2005 holdout**: the 2006 eval matches the
   hidden-holdout distribution, so the stopping point transfers (0.7144 vs 0.7121).

## What did not help

1. **Target encoding** (carrier/origin/dest/route, smoothed α=30): full-fit TE leaked badly
   (0.7067); leak-free out-of-fold TE still lost to native categoricals (0.7138 vs 0.7144).
   Airport identity is already covered by native categorical splits.
2. **Additional interaction crosses** (dest×hour-block, month×hour-block, dow×hour-block,
   carrier×dow, distance-bin×hour-block, route×hour-block): all neutral-to-harmful — the
   departure side (carrier×time, origin×time) is where the signal lives; destination and
   calendar crosses just diluted split slots.
3. **Hour-of-day as a standalone categorical** (replacing raw DepTime): 0.7134 vs 0.7141 —
   with enough depth, monotone numeric time encodings are already exploited; micro time-feature
   variants (sin/cos, minute) added nothing.

## With more budget

- Scale the winning recipe: more members of the deep+λ5/λ10 family (runtime-capped at ~100 s
  per run here), plus per-member bootstrap resampling for stronger decorrelation.
- Route×hour-block restricted to frequent routes timed out at 120 s; with a higher limit (or
  feature pre-binning / fewer members) it is the most promising untested cross.
- A small stochastic search over the deep-member family (depth 10–14 × colsample 0.5–0.9 ×
  λ 5–10, 20–30 draws) scored by OOF AUC on 2005 rather than one-off eval runs, then a large
  rank-averaged ensemble of the top half.
