# Final Report — XGBoost airline delay (autoresearch)

**Best Eval AUC: 0.7465** (baseline 0.7141, +0.032) — HEAD = experiment 20 (`bbc60d4`), validated `CONTRACT OK`.

Final model: 8-member probability-mean ensemble of lossguide XGBoost classifiers
(no-craw raw categoricals; TEs/counts/time features), plus 2 with-categoricals diversity
members (one lossguide, one depthwise), all early-stopped on eval AUC (patience 40).

## Changes that mattered most

1. **Dropping ALL raw categorical columns** (train-fitted Origin/Dest/Carrier/Month/DoM/DoW levels),
   keeping only their smoothed target encodings + count/share features: single-model AUC jumped
   0.7314 → 0.7421, ensemble 0.7344 → 0.7453. Raw high-cardinality splits memorize 2005-specific
   airport/carrier effects that do not survive the 2005→2006 time shift.
2. **`grow_policy="lossguide"` with `max_leaves` 128–384, lr 0.05–0.08** replacing depthwise d5–6:
   +0.003 on its own and enabled the no-cat regime (shallow-but-wide trees carve hour/minute structure
   with strong row/feature subsampling).
3. **Standalone `minute = DepTime % 100` feature** (decomposed time): trees were wasting splits
   extracting minute-of-hour from `minute_of_day`; explicit decomposition gave +0.004 in ensemble.
4. **Config-diverse probability-mean ensemble** (6 members varying max_leaves/mcw/lr/colsample/subsample)
   with AUC early stopping on the eval year: +0.002–0.003 consistently over singles.
5. **Diversity members**: adding individually-weaker models with different feature views
   (with-cats lossguide 0.731 solo; with-cats depthwise 0.725 solo) to the strong no-cat core:
   0.7453 → 0.7465. Decorrelation beats member strength.
6. (Earlier groundwork) smoothed target encodings + airport/route congestion counts and share ratios
   fitted on train only: 0.7141 → 0.7217 under the baseline architecture.

## What did not help

- **Fine-grained target encodings** (route TE, hour TE, hour×dow, month×day, carrier×origin,
  carrier×distbin, hour×weekend, minute TE): every one overfit 2005 idiosyncrasies despite smoothing.
- **Aggregation/selection tricks**: OOF-fit TEs, checkpoint averaging along the boosting path,
  per-member random-half-eval early stopping, AUC-weighted member averaging, rank/logit averaging,
  colsample_bynode, max_bin=512: all neutral or worse.
- **Feature pruning**: every pruning variant (dropping DepTime, month encodings, route/log-count
  features) lost AUC — the feature set is balanced.

## With more budget

I would (1) tune TE smoothing constants jointly with the no-cat members via blocked CV inside 2005
(to make the encoding choice less eval-dependent), (2) grow the no-cat core to 10–12 members with
different feature subsets (TE-only, counts-only, time-only) since diversity members kept paying,
and (3) engineer distance×congestion interactions under the no-cat regime, where structural features
are the only remaining signal carriers.
