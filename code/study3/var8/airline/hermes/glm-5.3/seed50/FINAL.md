# Final report — airline dep-delay XGBoost (autoresearch harness)

Best Eval AUC: **0.7358** (experiment #36, commit 5957f06), up from the 0.7141 baseline (+0.0217).
Validation: `./validate.sh` prints `CONTRACT OK`; AUC via `predict_proba` on raw frames reproduces 0.7358.
Budget used: 40/40 experiments, ~205 min wall clock of 230, ~3100 of 18000 CPU-seconds.

## What mattered most (in order of impact)

1. **Carrier x time-of-day interaction categoricals** (`Car_x_Hour`, `Car_x_Block`, 20x12/20x24 levels,
   native XGB categorical splits): +0.013 in one step — the single biggest gain. Airline-specific evening
   delay patterns are the dominant learnable signal.
2. **Smoothed target-rate numerics for interaction keys** (carrier x hour, carrier x 2h-block, carrier x
   30-min bin, coarse-origin x block/hour, carrier/hour x distance-band; empirical-Bayes smoothing M=20,
   fit on train only, inside `prepare()`): about +0.004 cumulative. Dense rates complement the categoricals.
3. **Capacity + regularization recipe**: 3000 trees, depth 8, lr 0.03, subsample/colsample 0.8,
   min_child_weight 5, lambda 2, early stopping on eval: +0.0025 over baseline 30-tree config.
4. **Congestion counts** (train-fitted flights per origin-hour, origin-block, dest-hour, hour): +0.0004.
5. **4-seed ensemble + feature weights** (2-3x weight on the proven interaction features): +0.0004.

## What did not help (all reverted)

1. **Route features in every form** — raw Origin_Dest categorical (−0.008), busy-routes-only categorical
   (−0.001), smoothed route rate (−0.002). 2005 routes do not transfer to 2006.
2. **Sparse/high-cardinality airport x time categoricals** (full Origin/Dest x block, −0.008) and
   **day-of-week x time-of-day interactions** (−0.006) — calendar/day effects don't generalize year-over-year.
3. **Seasonal rates** (month x carrier, month x origin): −0.004; also dest-side time rates, 3-way
   car x hour x distance rate, quantile-binned deptime curve, deeper trees (depth 10), heavier
   regularization, larger bagged ensembles (7 seeds x 0.7 sampling), and 5 seeds over 4.

## With more budget

I would (a) replace eval-based early stopping with time-aware CV inside 2005 (last months as validation)
to make stopping honest w.r.t. the hidden 2006+ holdout, since eval.csv currently influences both
keep/discard decisions and tree count — the main overfit risk left; (b) systematically sweep the
interaction-rate granularity (15/30/60/120-min bins x carrier/origin) with per-key adaptive smoothing
instead of the fixed M=20; (c) build a proper two-stage ensemble: K-fold OOF base models averaged with a
shallow, strongly-regularized XGB stacker on the rate features; and (d) explore count-based
congestion at finer time bins per airport, which showed the right sign but needs cross-year normalization
(2006 has different traffic volumes than 2005).
