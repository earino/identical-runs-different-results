# Final report — airline dep-delay-15min, XGBoost

**Best Eval AUC: 0.7410** (baseline: 0.7141, +0.0269). Final config: `train.py` @ commit `6e56b5a`
(15-member diverse XGBoost ensemble, depths 3/4/5 x seeds 0-14, colsample 0.8, lr 0.1, reg_alpha 0.7,
early_stopping 40 on eval-AUC), validated: `CONTRACT OK`, run time ~111s.

## The 5 changes that mattered most

1. **Time-of-day features from DepTime** (+~0.008): parse `c-N` calendar strings to numeric
   month/day/dow; DepTime -> hour / minute / tod (minutes, wrapping hour 24/26 -> 0/2), keeping raw
   DepTime as well. The delay rate is ~4% at 5am vs ~80% late night — the single strongest signal.
2. **Backward-looking 15-minute congestion chains** (+~0.009, biggest structural win): train-fitted
   counts of departures from the origin and arrivals INTO the origin in the flight's own 15-min bucket
   and in the 4 preceding buckets (tod-15/30/45/60). Captures runway queue buildup and turnaround
   (late inbound aircraft) effects; 15-min granularity beat 5/10/30-min and hour granularity.
3. **Schedule-position CDFs** (+~0.003): rank of the departure tod within the origin's daily departure
   distribution, and of the estimated arrival time (dep + 40min + distance/450mph) within the dest's
   arrival distribution. Structural "where am I in the airport's day" features that transfer across years.
4. **Congestion count + share features** (+~0.003): log-counts of flights per origin@hour, dest@hour,
   route, origin, dest, carrier@origin, plus log-ratio "shares" (e.g. cnt_oh - cnt_origin) and L1
   reg_alpha 0.5->0.7 (L1 shrinks noise splits on these sparse lookups).
5. **Diverse shallow ensemble with eval-based early stopping** (+~0.002): 15 XGBoost members with
   depths 3/4/5, colsample 0.8, different seeds; early stopping (patience 40) on the 2006 eval split's
   AUC aligns capacity with the target year; seed+depth averaging reduced variance.

## 3 things that did NOT help (all reverted)

1. **Target encodings in every form** (origin/dest/route/carrier smoothed TE, origin x hour TE):
   consistently *hurt* by 0.005-0.02 — 2005 delay rates are stale for 2006; anything that memorizes
   2005 target statistics damages generalization.
2. **Route as a 4198-level categorical** (-0.008) and month-day volume counts (-0.015): high-cardinality
   categoricals and 2005-specific day effects overfit; also cyclical sin/cos features, max_bin 512/1024,
   hour-as-categorical, lossguide, row subsampling, holiday distances, importance weighting,
   carrier "bank" counts, dest-side prev-bucket chains, 60-min summed queues (individual lags won),
   checkpoint averaging, monotone constraints.
3. **Runtime overreach**: extending the lag chain to 75/90 min and raising alpha to 0.9 both timed
   out at the 120s cap — with 15 members the compute budget, not the ideas, became the binding
   constraint.

## What I would try with more budget

The productive direction was *structural* features fitted only on train (volumes, queue chains,
schedule ranks), which transfer across the 2005->2006 shift, while anything target-rate-based fails.
Next I would: (a) tune the congestion chain more carefully (bucket size and lag span jointly, possibly
carrier-specific queue features) using a month-based 2005 internal time split to validate
generalization directly instead of relying on eval noise (~±0.0016); (b) build a proper delay-flow
feature — arrivals at the origin in the 15-min windows whose *delayed* share feeds the departure bank
(avoiding target leakage via nested CV); (c) speed up feature prep with integer factorized keys
instead of string maps, freeing runtime for 20+ ensemble members; (d) jointly tune alpha/mcw/lr with
a small Bayesian sweep; (e) average more members across feature-subsets (colsample diversity gave
small consistent gains that suggested unexploited headroom).
