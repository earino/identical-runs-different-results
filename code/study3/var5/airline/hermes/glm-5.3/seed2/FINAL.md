# Final report — airline dep-delay XGBoost (autoresearch harness edition)

**Best Eval AUC: 0.7519** (commit `1c78159`, experiment #35), up from the 0.7141 baseline (+0.0378).
`./validate.sh` prints `CONTRACT OK` and reproduces 0.7519 via `predict_proba` on raw rows (target column removed).

## What mattered most (in order of impact)

1. **Deep trees + low learning rate + early stopping** — the single biggest lever: max_depth 6→20 with lr 0.008–0.02,
   2000+ rounds and early stopping took 0.7141 → ~0.742 (experiments 2, 5–12). Depth alone explained most of it.
2. **Departure-time granularity** — DepTime binned into half-hour slots as a native categorical, plus hour-as-categorical
   before it (0.7472), plus cyclic sin/cos on the half-hour index (0.7519). Time-of-day is the strongest signal in the data
   and fine bins capture its non-linearity; the cyclic pair adds wrap-around continuity (23:30 ≈ 0:00).
3. **Numeric calendar features + cyclic encodings** — month/day/dow parsed from c-<n> strings with sin/cos variants
   (part of the first big jump, 0.7141 → 0.7214).
4. **3-model XGBoost bag** — averaging 3 models (different seeds, colsample 0.7/0.8) added ~+0.0007 robustly.
5. **Structural count/distance features** — log flight-frequency counts of Origin/Dest/Carrier (+0.0012) and route mean
   distance + deviation (+0.0005), all statistics fit on train only.

## What did not help (all reverted)

1. **Target encoding of any kind** (route/carrier/origin/dest/x-hour, smoothed m=20): 0.6696 — catastrophic. 2005 per-key
   delay rates do not transfer to 2006; the yearly regime shift makes label statistics poisonous.
2. **Interaction categoricals** (route, carrier×origin, carrier×hour, origin×hour, carrier×month): 0.7075–0.7325. Same
   disease — fine-grained 2005 co-occurrence patterns don't recur in 2006, and carrier×month was the worst (0.7273).
3. **Finer than half-hour time bins** (15-min: 0.7500, 20-min: timeout, 45-min: 0.7505) and seasonality aggregates
   (doy, month_day int, origin/dest mean distance: 0.7358), month-edge flags (0.7511), higher lr with esr 100 (0.7503),
   gamma 0.2 (no change), max_bin 256 (−0.0001).

## What I would try with more budget

The evidence says the model is at the ceiling of what per-flight static features can give; everything that failed shared one
root cause — the 2005→2006 distribution shift. I would attack that directly: (a) build features from *within-2006* unlabeled
structure only if allowed, otherwise domain-adapt via per-month/per-carrier prior correction using drift-robust statistics
(median ranks instead of means); (b) train a second model on a time-based subsample of train that better matches 2006's
carrier/month mix, and rank-average it with the current one; (c) enlarge the bag to 5+ diverse members (colsample 0.5–0.8,
max_depth 16–24) under a per-model time cap, since ensembling was the most reliable small gain; (d) sweep the half-hour
bin offset (0/10/20 min) and add per-slot carrier-share statistics computed as counts (not label statistics), which were
the only train-fit aggregates that transferred.

## Trace

40/40 experiments used, ~14,600s CPU of 18,000s, ~58 min wall. Log: `experiments.tsv`; best kept commit `1c78159`.
