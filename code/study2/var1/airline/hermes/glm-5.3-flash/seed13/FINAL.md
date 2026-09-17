# Final Report — airline delay XGBoost (autoresearch, scenario 2)

**Best Eval AUC: 0.7469** (baseline 0.7141, +0.033). Final commit `49da0a1`, 17 experiments used.

## Final model

20-member XGBoost bag: depth 10, gamma 5, alpha 1, mcw 10, subsample 0.9, over 5
lr/colsample families (800/0.05, 2400/0.02, 800/0.05+cs0.8, 2400/0.02+cs0.8, 1200/0.04+cs0.8)
x 4 seeds. Features: DayOfWeek, UniqueCarrier, Origin/Dest (rare airports collapsed to OTHER at
>=240 train rows), DepTime (15-min categorical slot + hour categorical + raw), Distance.

## Changes that mattered most

1. **Dropping year-drift-prone features** (Month entirely, DayOfMonth, route, airport counts,
   carrier-month interactions): 0.7164 -> 0.7448. Eval-2006 carrier/month rates differ sharply
   from 2005 (e.g. AS 0.64->0.52, FL 0.62->0.54); anything fitting those cells overfit the drift.
   This was the single biggest lever, and it *removed* code.
2. **Departure time-of-day engineering**: DepTime minutes-of-day as 15-min categorical bins
   (depbin96, +0.005), plus 30-min bins and hour categorical. Late-night/early-morning flights
   are the strongest single signal in the data.
3. **Strong regularization enabling depth 10** (gamma 3-5, alpha 1, cs 0.5): 0.7288 -> 0.7381.
   With drift-noisy features, gamma/alpha pruning beat growing capacity naively.
4. **Seed/config bagging** (5 -> 20 members with lr and colsample diversity): ~+0.008 total,
   monotone and robust — the most reliable gains in the whole run.
5. **Rare-airport collapse to OTHER** (>=240 train rows): +0.0004 and fewer categories.

## Things that did NOT help

1. **OOF target encoding** of Origin/Dest/carrier (and any smoothed-rate numeric variant) —
   always below the categorical baseline; the drift poisons learned rate maps.
2. **Composite interaction categories** (depbin x DOW, hour x month, carrier x bin) — large
   eval-time losses (up to -0.02); they re-introduce the year-drift the feature-dropping removed.
3. **Cyclical sin/cos time encodings, raw-minutes numerics, depth 4-6 shallow configs,
   lossguide, 5-min bins, per-node colsample, row-bootstrap bagging** — all neutral or negative.

## What I would try with more budget

A leakage-safe year-debiasing scheme: fit per-carrier/per-airport effects on 2005, then shrink
them toward the 2006 marginal (estimated label-free) instead of toward the pooled mean — the
train/eval rate tables show the residual carrier effects are large but partly year-specific.
Second: a proper greedy forward-selection of feature subsets against the drift (every candidate
interaction needs both CV and eval confirmation, which the CPU budget no longer allowed).
