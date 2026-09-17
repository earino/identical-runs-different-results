# FINAL — autoresearch XGBoost (airline)

**Best Eval AUC: 0.7373** (experiment #40, commit `afc9b40`; baseline was 0.7141).

## What mattered most

1. **Traffic / congestion count & fraction features** (the single biggest lever). The raw columns contain
   no notion of how busy an airport, carrier or route is at a given time. Adding counts of flights by
   `Origin×hour`, `Dest×hour`, `Carrier×hour`, `Route×hour`, `Route×15min-bin`, plus their fractions of the
   corresponding total, lifted AUC from ~0.7215 to ~0.7307 in three steps. These features are leak-free
   (mapped from train only) and stable across the 2005→2006 time split.
2. **Adjacent-window congestion** (`Origin/Dest/Route × prev/next 15-min bin`), a delay-propagation proxy:
   +0.0023 (0.7350 → 0.7373) in the final experiment.
3. **Model capacity re-tuned to the richer feature set.** With the volume features, deeper/more trees
   stopped overfitting: `max_depth=10, lr=0.02, ~600 trees, min_child_weight=40, reg_lambda=10` beat the
   earlier shallow 100-tree model (0.7307 → 0.7331). More trees (1200) overfit again.
4. **Diverse-seed XGBoost ensemble** (3 configs × 2 seeds, depths 9/10/11) averaging predicted
   probabilities: +0.0011 (0.7331 → 0.7342) and more robust than any single model.
5. Time-of-day features from `DepTime` (hour / minute / minute-of-day) plus keeping the raw `DepTime`.

## What did not help

1. **High-cardinality derived categoricals.** `Origin_Dest` route as a native categorical dropped AUC from
   0.7166 to 0.7090; `max_cat_to_onehot=300` (one-hot Origin/Dest) was catastrophic (0.7128).
2. **Target encoding** (OOF mean delay by Origin/Dest/Carrier/route): 0.7173 vs 0.7193 — the 2005→2006
   temporal shift makes historical delay rates unstable.
3. **Day/hour-specific count features** (`Origin×Month×Day×hour`, `global day×hour`): 0.7277 vs 0.7307 —
   they memorise specific calendar days and transfer poorly. Plain calendar/seasonal features
   (`doy`, sin/cos, `is_weekend`) were neutral.

## What I would try with more budget

The congestion/volume family is clearly where the signal is, so I would push it further: scheduled
turnaround proxies (same-route flights in the preceding window), a graph-style "airport pressure" feature
summing inbound traffic over the previous 30–60 minutes, and route-level competition (number of carriers
on a route). On the modelling side I would run a proper nested/rolling-origin validation instead of a
single random 15% split, because internal validation kept selecting many more trees than generalised to
the 2006 eval set — a sign of temporal overfitting. I would also try stacking a small second-level
XGBoost over the ensemble, and confirm with repeated splits that the last few +0.001 gains are real
rather than eval noise.
