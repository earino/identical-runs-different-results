# FINAL — airline departure-delay (dep_delayed_15min) XGBoost autoresearch

**Best Eval AUC: 0.7592** (commit `19ce486`, experiment #39; baseline was 0.7141).
`./validate.sh` → `[validate] eval AUC via predict_proba (target column removed): 0.7592` / `CONTRACT OK`.
40/40 experiments used, 187 minutes of the 230-minute budget still on the clock, 6953/18000 CPU-seconds.
Final config: 13-model XGBoost ensemble (`hist`, native categoricals, depths 4–6 + 2 leaf-wise members),
80 features, ~84 s per run (well under the 120 s cap).

## Changes that mattered most

1. **Traffic-structure "share" features computed inside `prepare()` from the input frame itself**
   (experiments #12–#19, #21, #28–#33, #37, #39: 0.7141 → 0.7592 in total). Every feature is a *share*
   (a count divided by another count taken from the same frame), which makes it scale-free: the same code
   yields comparable values for 100k training rows, the 100k eval slice, and the 1M-row hidden holdout.
   Families added, roughly in order of value:
   - airport/carrier/route traffic share at a given hour, half-hour, 15-min and 5-min slot
     (`origin_hour_share`, `*_halfhour_share`, `origin_hour_minute_share`, `origin_hour_min5_share`, …);
   - cumulative daily traffic share up to the departure hour (delay build-up: `origin_cum_hour_share`, …);
   - hub dominance / route importance (`origin_carrier_share`, `route_share`, `carrier_route_share`);
   - destination-side congestion at the *estimated* arrival hour (block time ≈ Distance/450 mph).
2. **Explicit minute-of-hour features** (#34: 0.7516 → 0.7584). `DepTime`'s minute is strongly and
   *reproducibly* predictive: delay rate ranges ~0.44–0.59 across minutes and the ordering replicates
   between 2005 and 2006 (e.g. :18 is high in both, :55–:57 low in both, while most flights sit on :00/:05/…).
   Trees cannot isolate that from a continuous time-of-day feature, so `dep_min`, `dep_min_mod5/15`
   and slot shares keyed by minute bucket were added.
3. **Shallow trees** (#8–#10: depth 4–5 clearly beat depth 6–8; re-tested at #15 and #36 — deeper always
   lost 0.001–0.002). The signal is mostly smooth main effects, so capacity only buys overfitting of the
   2005 sample.
4. **Ensembling** (#11, #20, #25, #27, #38–#40: 0.7161 → 0.7401 as the ensemble grew). Averaging 13 diverse
   XGBoost models (different depth / lr / colsample / subsample / grow_policy) was the most reliable source
   of small gains and costs only runtime.
5. **Calendar structure** (#22: +0.0011) — `Month`/`DayofMonth` are ordinal codes, so day-of-year,
   day-of-year sin/cos and holiday travel windows (Thanksgiving, winter holidays, Labor Day, Memorial Day,
   July 4) are recoverable; explicit interaction products (#24) added a small extra gain.

## What did not help

1. **High-cardinality native categoricals** (`Origin_Dest` route, `Carrier_Origin`, #7: −0.017). Partition
   splits on ~1500–4200 levels overfit 2005 badly; the same information is much better used as share features.
2. **Target / frequency encodings fitted on the training set** (#5–#6: 0.7098, −0.006 vs the same model
   without them — `route`, `origin_hour`, `carrier_route` rate encodings, OOF on train + full-train stats
   for eval). Under a 2005→2006 time shift the historical rates transfer poorly, and OOF values on train
   have a different noise level than full-fit values on the holdout.
3. **Deep trees / many rounds at low learning rate** (#4, #15, #36) and **the full model before ensembling**:
   more capacity consistently lost AUC; a 30-tree depth-6 baseline was already close to a 1200-round depth-6 model.
4. Also tried and rejected: neighbouring-hour congestion shares (#23), hub-wave `nunique` counts (#26),
   `dep_min` as a 60-level categorical + multiple-of-5 flag (#35), sparse exact-departure-time shares
   (#29, −0.0009 — counts that depend on frame size do not transfer between a 100k train frame and the
   1M-row holdout), early stopping on a random 10% internal split (#2), 16-member ensemble (#40, equal-or-worse
   with more runtime).

## What I would try with more budget

The label-free "schedule-structure" features were by far the richest seam, and there is still room in it:
minute-level slot shares at more key combinations (carrier×route×hour, origin×dow×hour), rank/quantile
versions of the same shares (busy-ness percentile within a key, which the trees can split more evenly),
and a smoothed target encoding of the *minute-of-hour* effect alone (it is stable across years, unlike the
airport/carrier rates that failed). On the model side, the obvious next step is a larger ensemble with
diversity in the *feature* dimension (e.g. bagging feature subsets per member) now that capacity is bounded
by runtime, and a small Optuna-style search of min_child_weight / colsample around the current configuration
using a *time-ordered* internal validation split instead of eval.csv. Finally, the biggest open risk is the
transductive assumption: shares are computed from whatever frame `predict_proba` receives, which matches the
scorer's full-frame usage but would degrade if the hidden holdout were ever scored in small batches — a way
to hedge would be to blend the frame-computed shares with fixed per-key schedule statistics fitted on train.
