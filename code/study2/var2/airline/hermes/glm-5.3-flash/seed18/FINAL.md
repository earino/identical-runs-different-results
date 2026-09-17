# Final report

Best Eval AUC: **0.7319** (experiments #35/#36; baseline was 0.7141). Final model: 5-member
XGBoost bag (diverse depth/colsample/subsample/row-subsets) at lr 0.04, cap 1000 trees,
max_bin 512, early stopping on a last-20% time split with 1.1x refit on all rows.

## Changes that mattered most

1. **Congestion counts + hourly shares** (#10, #12, +0.0073 then +0.0041): log1p flight counts
   per (Origin|DepHour), (Dest|DepHour), (Carrier|DepHour) and per airport/carrier, plus
   share-of-traffic ratios (joint/marginal). Delay is driven by schedule pressure; these dense,
   target-free encodings were the single biggest lever.
2. **Time-of-day decoding** (#2, +0.0032): DepHour numeric (including the hour>=24 quirk),
   sin/cos of minutes-since-midnight. Raw hhmm as an integer is nearly unusable to trees.
3. **Big model + early stopping on a time split + refit** (#3, +0.0008): 1500-tree lr-0.05
   regime instead of 30 trees at lr 0.1; ES iteration count measured on the last 20% of train,
   then refit on all rows.
4. **Ensembling** (#15/#17/#18/#22/#28, +0.0066 cumulative): 3-seed mean -> 5-fold row-subset
   bagging -> per-member max_depth/colsample_bytree/colsample_bynode/subsample diversity.
   Consistent, additive gains up the whole ladder.
5. **Capacity/precision tuning at the end** (#24/#25/#29/#31/#35, +0.0039 cumulative):
   dep_min numeric, hour>=24 and minute==0 flags, lr 0.04, max_bin 512, and a 1000-tree cap
   (smaller models stopped earlier and generalized better; also bought back wall-clock).

## Things that did not help

- **Target encoding** of carrier/origin/dest/route (#4, -0.0018): even smoothed and fit on train
  only, it underperformed plain categoricals + counts here.
- **High-cardinality route as a native categorical and sparse interaction counts** (#6/#11/#14,
  -0.009 to -0.017): 4198 route levels fragment trees; sparse keys (route x hour, carrier x
  route, day x hour) overfit and can even time out.
- **Calendar recodes and month weights**: calendar-as-categorical (#8), day-of-year (#19),
  calendar density counts (#20), origin/dest x dow counts (#38), recency sample weights (#26,
  tie) all failed to beat the simpler setup; lr/depth up-tunes (#5, #16, #30) and extras like
  min_child_weight (#21), max_bin 1024 (#33), K>=6 bags (#23, #32, #40), log-distance (#39)
  also regressed or tied.

## With more budget

I would run proper CV-based ES (averaged over folds rather than one split), then a larger
diversity ensemble (more members at cap 1200-1500 with max_bin 512), revisit smoothed
target encoding with CV-fit out-of-fold values (the leakage-free variant I never tested),
and engineer richer flight-network features (inbound delay pressure proxies: fraction of
late-arriving feed at the destination airport per hour, built from train-only delay rates
with heavy smoothing).
