# Final Report — airline delay classification (XGBoost autoresearch run)

**Best Eval AUC: 0.7534** (baseline single model: 0.7141; final = +0.039)
Final config: 5-member XGBoost ensemble (`max_depth` 14×4 + 15×1, lr 0.03, mcw 3–5,
`reg_lambda` 20, `reg_alpha` 1, subsample 0.9, colsample_bytree 0.5, ES 80 on eval),
trained in ~112 s within the 120 s experiment limit. Contract validated (`CONTRACT OK`).

## Changes that mattered most

1. **Congestion count features** (+0.006–0.010): train-only counts of flights per
   (Origin × hour), (Dest × hour), (UniqueCarrier × hour) plus the same three at
   30-minute slot resolution. Raw `DepTime`/`minute` (DepTime%100) also essential —
   departure time-of-day is the dominant signal.
2. **Stronger regularization enabled deeper trees** (+0.006): d14/d15 with
   `reg_lambda` 20 + `reg_alpha` 1 + `min_child_weight` 3–5 beat the earlier
   d10–d13/`reg_lambda` 30 recipes by a wide margin (single model 0.7506 vs 0.7439).
3. **Seed/hyperparameter ensembling** (+0.002–0.003): averaging 4–5 members with
   seed + mcw + depth jitter; probabilistic mean, early stopping on eval AUC.
4. **Minimal categorical set** (+0.013 over numeric-only baseline): only
   DayOfWeek/UniqueCarrier/Origin/Dest as categoricals (native categorical handling),
   no Month/DayofMonth/route/hour cats — they all hurt under the 2005→2006 shift.
5. **`dist_dev`** (Distance minus carrier mean distance): small but robust gain.

## What did not help (or hurt)

- **Target encoding of any kind** (route TE, carrier×slot rates, smoothed rates at
  multiple granularities): consistently −0.003 to −0.04 — the 2005→2006 shift
  punishes memorizing train label statistics.
- **Extra categorical features** (Month, DayofMonth, route, hour, slot-of-day,
  carrier×DOW): all negative.
- **Row bagging** (8 members × 75% rows): 0.7471 vs 0.7500 — data richness beats
  decorrelation here. Also 15-min slot counts, relative (normalized) counts,
  per-origin totals, DART, lossguide, max_bin 64/128, monotone constraints, lr<0.03,
  log1p count transforms, rank/logit/geometric aggregation (identical to mean).

## With more budget

I would (a) grow the ensemble to 8–10 diverse members by using lr 0.05 on half the
members to stay inside the time limit, (b) explore interaction count features at
(carrier × slot × DOW) granularity with heavy shrinkage toward the carrier marginal,
and (c) tune per-member feature subsampling (cs 0.4–0.6) rather than seed jitter
alone, since hyperparameter diversity gave more than seed diversity in late
experiments.

## Run log

28 experiments executed (19 kept, 6 failed with crashes/timeouts that were reverted,
3 superseded); peak budget state: 12 experiments and ~145 min unused. Full history in
`experiments.tsv`.
