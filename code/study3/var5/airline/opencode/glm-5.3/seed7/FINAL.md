# Final report — airline delay XGBoost benchmark

**Best Eval AUC: 0.7312** (experiment #39, commit f2c5eb2), up from the 0.7141 baseline (+0.0171).
`validate.sh` prints `CONTRACT OK` and reproduces AUC 0.7312 through `predict_proba`.

## What mattered most (in order of impact)

1. **Dropping noisy calendar encodings** — removed DayofMonth entirely and the *categorical* forms of
   Month (keeping smooth `Month_i` ints): +0.003. Category-identity splits memorize 2005-specific
   calendar noise; ordered ints force smooth seasonal structure that transfers to 2006.
2. **Traffic-density features (target-free)** — log flight counts per Origin, Dest, Carrier, hour, and
   their pairwise combos (origin-hour, dest-hour, carrier-hour, route), computed on train only and
   mapped onto any new rows: +0.001-0.002. Congestion proxies are persistent year over year.
3. **L1 regularization ramp (reg_alpha 0→1→3→6)** — the single biggest late-stage lever, +0.007 over
   lambda-only. Shrinking spurious leaf weights directly attacks 2005→2006 memorization.
4. **Split-level feature sampling (colsample_bynode=0.6)** on top of per-member colsample diversity: +0.001.
5. **Diverse 7-member seed ensemble** (varying colsample_bytree 0.6–1.0, averaging probabilities) plus
   a regularized core (depth 7, mcw 50, lambda 20, subsample 0.8, 400 trees @ lr 0.05): +0.002 cumulatively.
   Also kept: both time scales (raw DepTime hhmm + dep_min + sin/cos of day) and Distance.

## What did NOT help (all reverted)

- **Target encodings of any kind** (carrier/origin/dest/route/origin-hour, even smoothed hour-only):
  0.7066 vs 0.7150 — they hard-code 2005 label noise into features.
- **Route identity as a native categorical**: 0.7030 — the worst single result; route-level effects
  are year-specific poison for transfer.
- **rank:pairwise objective** with one 100k-row group: collapsed to 0.5000 (and slow).
- Deeper trees without density features, 600+ trees, lower lr with more rounds, max_bin=64,
  monotone constraint on dep_min, dropping Distance / raw DepTime / sin-cos — all worse or equal.

## What I'd try with more budget

The L1 curve (1→3→6 up, 12 down) peaked near alpha 6-9, so I'd fine-sweep alpha in [5,10] jointly with
lambda, and re-tune min_child_weight at the final config since regularization changed the leaf economics
late in the run. Second, a larger *diverse* ensemble (12-15 members varying depth/subsample/colsample
and feature subsets — e.g., members without Origin/Dest) now that members are individually noisier but
less correlated. Third, OOF-based stacking of the members' predictions via a tiny XGBoost to learn the
blend instead of a flat mean. Finally, seasonal-interaction features (Month x dep-hour curves) as
smooth numeric summaries — plain splits approximate them, but explicit cross-year-stable summaries
(e.g., per-hour-bucket mean DepDelay statistics aggregated to monthly curves) might transfer.
