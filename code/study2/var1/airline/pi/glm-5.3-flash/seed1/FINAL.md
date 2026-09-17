# Final report — airline delay AUC (XGBoost)

**Best Eval AUC: 0.7285** (baseline: 0.7141, +0.0144). Hidden holdout is scored via `predict_proba(df)`
(contract validated: `CONTRACT OK`, 0.7285 reproduced with the target column removed).

## Final architecture

Ensemble of 6 XGBoost models (`hist`, native categoricals, `max_bin=512`), probability-averaged:

| # | config | view |
|---|--------|------|
| 1 | depth10 n800 msw20 λ5 | full |
| 2 | depth12 n700 msw10 λ3 seed777 | full |
| 3 | depth12 n700 msw10 λ3 seed888 | **Route dropped** |
| 4 | depth10 n800 msw20 λ5 seed889 | **Route dropped** |
| 5 | depth12 n600 msw30 sub.8 col.6 λ5 | full |
| 6 | depth12 n700 msw10 λ3 seed4242 | **months ≥ 7/2005 only** |

All members use month-based recency sample weights (Jan=1.0 … Dec=1.88). Features: parsed
month/day/dow ints, dep hour/min/15-min-slot + sin/cos of time-of-day, log-distance, carrier,
Origin, Dest, Route as native categoricals. Target-encoding maps and all statistics are fit on
train only, inside the `prepare()` path (validated on a target-less frame).

## Changes that mattered most

1. **Ensembling diverse XGBoost configs** (exp 9): 30-tree single model 0.7141 → 3-model 0.7188
   (+0.0044). Variance reduction across depth/subsample/seed transfers across the 2005→2006 shift.
2. **Information-constrained members** (exp 35, 39): a twin of the strongest config trained
   *without* the Route column (must reconstruct Origin×Dest structure itself) gave +0.0021; a
   second Route-dropped member +0.0011. Constraint diversity decorrelates errors far better than
   seed noise.
3. **Recency sample weighting** (exp 22-23): upweighting late-2005 months (eval is 2006) gave
   +0.0005-0.0007 — direct drift adaptation.
4. **Parsed time features + bigger early-stopped model** (exp 3): dep hour/min, cyclic encodings,
   Route categorical, ~2000 trees (+0.0003 solo, but the foundation for everything after).
5. **max_bin 512 + numeric 15-min dep_slot** (exp 20, 28): +0.0003 each.

## What did not help

1. **Target encoding** (solo, in-sample, or even as an ensemble view): -0.001 to -0.009. Group
   delay rates from 2005 do not transfer to 2006; XGBoost's native categorical splits are better.
2. **Congestion/frequency features** (flight counts per origin/dest/route/origin-hour): in-2005
   val AUC rose (0.7647→0.7744) but 2006 eval fell — they memorize 2005 traffic volumes.
3. **Stronger single-model regularization** (depth 6, msw 20-25, heavy λ): 0.7126-0.7236 —
   underfits or shifts the ensemble off its sweet spot; also colsample_bynode, lossguide members,
   cat_smooth, time-free / coarse-time constrained members: all ≤ best.

## Theory

The 2005→2006 year shift dominates: in-train validation (0.75-0.77) massively overestimates 2006
transfer (0.71-0.73), and anything that fits 2005-specific rates or volumes (TE, congestion
counts) actively hurts. What transfers: stable scheduling structure (time-of-day cascade, season,
carrier/airport identity), which XGBoost captures from native categoricals + parsed time features.
Ensemble members therefore help only when they differ in *structure* (depth, feature constraints,
temporal focus), not in noise (seed).

## With more budget

- A 4th/5th Route-dropped member (the 3rd attempt timed out at 122s; trim members to fit) — the
  clearest remaining signal.
- Member-level row bootstrap + per-member recency-focus sweep (months ≥ 6/9).
- OOF-stacked weighting of members via a tiny XGBoost meta-learner (only if CV'd inside 2005).
- Feature view: carrier×slot constrained member; drop-column importance ablation per member.
