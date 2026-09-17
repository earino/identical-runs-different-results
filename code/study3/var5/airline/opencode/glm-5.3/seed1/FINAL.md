# Final Report — airline departure-delay AUC

**Best Eval AUC: 0.7485** (baseline 0.7141, +0.0344). Kept commits: exp #15/#16
(8-member deep ensemble, `1e46e0b`). Validation: `CONTRACT OK` (0.7485 via
`predict_proba` with target column removed).

## Changes that mattered most
1. **Fine time-of-day bins x carrier interaction** as categorical features
   (`b15_carr`, `b30_carr`, binkey = `dep//100*(60/m)+(dep%100)//m`): 0.7141 -> 0.7376.
   The single strongest signal; scheduled-time delay profiles are carrier-specific.
2. **Time-jitter augmentation**: train rows plus +/-8-minute shifted copies (weight 0.5)
   smooth the time-bin boundaries and the numeric time features: ~+0.004.
3. **Deep members** (`max_depth=10`, 250-400 trees, lr 0.04-0.05) beat shallow ones
   (d6/d8) as ensemble members and are cheap: 0.7473 -> 0.7485.
4. **Granularity/seed/config-diverse XGBoost ensemble** with per-binset cached
   augmentation (members alternate bin sets (15,30)/(20,60)): 0.7408 -> 0.7485.
5. **Regularization + cyclic features**: reg_alpha 0.5, reg_lambda 10,
   colsample_bynode 0.5, min_child_weight 20; sin/cos of minute-of-day,
   operational-day shift (day starts ~4:30am, shifted dow for red-eyes),
   day-of-year and month-cyclic numerics.

## Things that did not help
1. Route interaction (`Origin`x`Dest`) and hour x {dow, month, origin, dest} crossings - all hurt.
2. Target/frequency encoding, per-date flight counts, holiday buckets, recency weights - hurt or neutral.
3. Half-row jitter (jitter only a random half of rows): -0.0025; a 4th seed added nothing
   (seed axis saturated at 2-3); dart/lossguide/interaction_constraints/early stopping - no gain.

## With more budget
The plateau at ~0.7485 was compute-bound (120 s cap), not idea-bound. Singles showed
further headroom: 550/10/lr.03 scored 0.7480 and 400/10 with `max_bin=512` scored
0.7478, but the paired ensemble timed out; with more CPU I would build an
ensemble from exactly those two members (plus a d12 variant), explore
carrier-adaptive time granularity, per-member bagging axes (subsample/colsample),
and out-of-fold stacking of member predictions. On the data side, the 2005->2006
time shift rewards capacity control: I would revisit sample weighting toward
late-2005 rows as a gentler alternative to the capacity cuts that consistently
helped.
