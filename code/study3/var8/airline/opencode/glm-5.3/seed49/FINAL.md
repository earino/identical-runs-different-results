# Final report — airline dep-delay XGBoost benchmark

**Best Eval AUC: 0.7202** (baseline: 0.7141, +0.0061). Budget: 40/40 experiments used.
Final model validated: `CONTRACT OK`, `predict_proba` reproduces 0.7202 on eval rows.

## Final train.py in one sentence
A 200-member bagged ensemble of tiny XGBoost models (30 trees, lr 0.1) over a 72-way diversity
grid (max_depth 3-10 x subsample 0.7/0.8/0.9 x colsample 0.7/0.8/0.9), trained on the raw
feature set minus DayofMonth, with a linear recency ramp (Jan 2005 weight 1.0 -> Dec 2.0),
predicting by simple probability averaging.

## The 5 changes that mattered most
1. **Bagged ensemble of the baseline config** (25 -> 60 members): 0.7141 -> 0.7162. The
   single biggest jump. With 100k rows and a 2005->2006 year shift, averaging many
   high-variance small models is worth far more than any single-model tuning.
2. **Config-space diversity inside the ensemble** — mixing max_depth 5/6/7 and
   subsample/colsample 0.7-0.9 per member: 0.7162 -> 0.7170. Extending the depth grid to
   4-9 then 3-10 kept paying (0.7188 -> 0.7197 -> 0.7198); depth diversity is the axis
   that mattered most.
3. **Recency weighting**: linear ramp over 2005 months (Jan 1.0 -> Dec 2.0): +0.0003-0.0004.
   Confirms year-over-year drift in carrier/airport delay rates; late-2005 is closer to
   the 2006 distribution. A steeper ramp (x2.5 / Q4x3) overshot and was worse.
4. **Dropping DayofMonth**: +0.0005. Day-of-month is 2005-weather noise that costs
   capacity in 30-tree members; every added feature otherwise diluted them.
5. **More members** (60 -> 120 -> 200): +0.0003 total. Monotone, cheap, safe.

## What did NOT help (all reverted)
- **More capacity in any single model**: 45/100/150 trees, depth 7-8, or early stopping
  all scored 0.708-0.714. The 30-tree/depth-6/lr-0.1 point is a sharp optimum; anything
  bigger memorizes 2005-specific noise that does not transfer to 2006.
- **All target encoding** (smoothed, OOF or full-train): carrier/origin/dest/route and
  carrier/origin/dest x 3h-bin interactions — despite those group stats being very stable
  year-over-year (corr 0.91-0.95), every TE variant *lost* 0.003-0.006. Same for hour
  features (numeric hour/minute, hour-as-categorical, te_hour) and high-cardinality
  interaction cats (route, carrier-origin, carrier-dest): -0.002 to -0.007.
  Tiny models cannot afford any feature that is not pulling weight.
- **Monotone constraint** on time-of-day (delay rate rises through the day): -0.004.
  The 0-4am "already-delayed flights scheduled past midnight" pattern breaks monotonicity.
- **One-hot encoding small categoricals** (max_cat_to_onehot=32): -0.003; partition
  splits are better here. **min_child_weight=10**: -0.003 (defaults already optimal).
- **Deep/fraction-diverse members beyond the sweet spot**: subsample/colsample 0.5-0.65
  members are too weak and drag the average; n_estimators mix (20/30/45) and lr mix
  (0.08/0.1/0.13) were neutral-to-worse — only *depth* diversity paid.

## What I would try with more budget
The plateau at ~0.720 with raw features looks like an information ceiling for 100k
training rows under a year shift. I would next (a) build the ensemble with per-member
early stopping on a *time-based* split (train Jan-Sep 2005, validate Oct-Dec 2005) to see
if members can safely grow beyond 30 trees when stopping is aligned with the year shift;
(b) test gradient-boosted "snapshots" (save every 10th round of one 300-tree run and
average them — a cheap 30x-ensemble that reuses one training pass); (c) revisit the
stable carrier x hour-bin statistics as *offsets* rather than features (fit one tree on
the residual after subtracting the smoothed rate); and (d) calibrate member weights in
the average by a small internal time-based validation. I would also check whether
December 2005 should be downweighted (holiday-weather extremes) while late-year is
upweighted — the two effects fight each other inside a plain recency ramp.
