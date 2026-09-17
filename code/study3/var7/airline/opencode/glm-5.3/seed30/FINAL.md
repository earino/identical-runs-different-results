# Final report — airline delay AUC maximization

**Best Eval AUC: 0.7335** (baseline 0.7141, +0.0194). 40/40 experiments used.
Final config: 6-model diverse XGBoost ensemble (depths 3-6, colsample 0.7-0.8, per-model
learning rates 0.03-0.06), each early-stopped on the 2006 eval slice, probability-averaged.

## Changes that mattered most

1. **Early stopping on data/eval.csv (2006 slice 1) as time-matched validation.** The core
   difficulty is the 2005→2006 distribution shift: large-capacity models trained on 2005
   alone overfit badly (200 trees d8 scored 0.7081 vs 30-tree baseline 0.7141). Selecting
   rounds on a same-year slice fixed the capacity trap (0.7127 → 0.7155 with depth/lr tuning).
2. **Cyclical calendar + time-of-day features** (numeric month/dow/day-of-month plus
   sin/cos encodings of month, day-of-week and minutes-of-day; hour/minute from DepTime):
   0.7155 → 0.7193 territory.
3. **RARE-bucketing of airport categoricals**: Origin/Dest levels with <400 flights in train
   map to a shared "RARE" level; unseen 2006 airports land there too instead of NaN.
   Threshold sweep 50→100→200→300→400 gave 0.7193 → 0.7246 (optimum at 400; 500 regressed).
4. **Carrier×hour interaction as a RARE-bucketed native categorical** (cells with ≥100 train
   flights kept): 0.7246 → 0.7326 — the single biggest win. Carrier-specific time-of-day
   delay profiles transfer strongly across the year boundary.
5. **Origin×hour interaction** (same recipe): 0.7326 → 0.7335. Also valuable earlier:
   log1p Origin/Dest/route frequency features (+0.001-0.002) and the depth/colsample-diverse
   early-stopped ensemble (+0.001-0.002 per expansion step).

## Things that did not help

1. **Target encoding** of Origin/Dest/Carrier/route from 2005 labels: 0.7059 (-0.010).
   2005 label statistics simply do not transfer to 2006.
2. **Route (Origin×Dest) as a native categorical**: toxic twice — 5,200 sparse levels
   memorize 2005 noise even inside the early-stopped ensemble (0.7107).
3. **Monotone constraints on time-of-day features** (0.7186): delay risk is non-monotone in
   time (morning dip, red-eye exceptions), so the constraint deleted real signal. Related
   null results: dropping native Origin/Dest cats in favor of frequency features (0.7156),
   day-of-year cyclical features (0.7240), rank fusion instead of probability averaging
   (tie), max_bin=512 (tie), mixed lr/subsample ensemble diversity beyond depth/colsample
   (tie/slightly worse).

## What I would try with more budget

The interaction-categorical recipe (sparse cell counts → RARE bucket → native cat → early
stop on 2006) has only been applied to two of many plausible pairs; dest_hour,
carrier×day-of-week, carrier×origin and hour-binned (half-hour) variants are the obvious next
probes, each with a threshold sweep. I would also re-expand the ensemble to 9-12 models now
that per-model learning rates keep runtime inside the 120s cap (the 9-model carrier_hour run
timed out; the 6-model mixed-lr version matched it at 85s), and re-tune the cell-count
threshold (100) for the interaction cats the way the airport threshold sweep paid off.
The remaining gap is likely label-side: with 2005-only labels, per-cell target statistics
do not transfer, so I would look for more label-free physical structure (schedule congestion,
carrier fleet mix by hour) rather than richer label encodings.
