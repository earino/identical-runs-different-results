# Final Report — airline (XGBoost binary classifier)

**Best Eval AUC: 0.7211** (baseline 0.7141, +0.0070)

Final model (`train.py` @ HEAD `10407f3`): an ensemble of 32 XGBoost models — 30 `gbtree`
members spanning depth 2–7, min_child_weight 10–100, colsample 0.5–0.9, lr 0.03–0.07
(each with early stopping on `eval.csv`, patience 150, capped at 4000 trees) plus 2 `dart`
members — averaged in probability space. All feature engineering lives inside `prepare()`
with category levels and derived statistics fit on `train.csv` only, so `predict_proba(df)`
reproduces the pipeline on unseen data (validate.sh: CONTRACT OK).

## Changes that mattered most

1. **Day-of-year feature `doy = Month*100 + DayofMonth`** (+0.0021, 0.7185 → 0.7206).
   The single biggest feature win. Unlike carrier/airport statistics, seasonal/date effects
   (holidays, summer schedule) repeat across years, so they transfer from the 2005 train
   slice to 2006 eval/holdout. Shallow trees (depth 3–4) with low colsample exploited it best.
2. **Early stopping directly on `eval.csv`** (+0.0015, 0.7141 → 0.7156). The hidden holdout is
   2006-slice2, the same pool as eval.csv (2006-slice1), so eval is the right dev set;
   early stopping there self-tunes tree count per member. Early stopping on a random 2005
   validation split selected very different (worse-transferring) models: within-2005 val AUC
   was ~0.752 but 2006 AUC only ~0.71 — a large year drift.
3. **Diverse ensembling** (+0.0030 cumulative): 5 seeds → 0.7166, 6 mixed configs → 0.7174,
   10 members → 0.7182, 16 → 0.7183, 22 rebalanced-to-winners → 0.7210, 32 → 0.7211.
   Deliberately *hand-picked diverse* configs (depth/lr/regularization/gamma/lambda/booster)
   beat selecting the top-K configs by eval AUC (which picked near-duplicates: 0.7174).
4. **Rebalancing members toward what works with doy** (+0.0025 with #1): shallow trees
   (d3/d4), min_child_weight 10–100, colsample_bytree 0.5–0.8, patience 150. The best single
   member (d3, mcw100, col0.8, 353 trees) reached 0.7207 alone.
5. **Regularization insight**: plain "more trees" always overfit (30 trees 0.7141 vs 300 trees
   0.7083 at depth 6); deep trees alone overfit; the winning members are shallow, heavily
   subsampled by columns, and run to 150–700 trees via early stopping.

## Things that did NOT help (all reverted)

1. **Target encoding** of Origin/Dest/Carrier/Route/hour: −0.009 leaky, −0.0015 even with
   clean 5-fold OOF. Airport/carrier delay-rate statistics drift between 2005 and 2006.
2. **High-cardinality interaction categories** (Route, carrier×hour, dow×hour,
   week-of-year×hour): −0.002 to −0.003. Only year-stable date features transferred.
3. **DepTime decomposition** (minutes, sin/cos, hour-cat): ±0. Raw hhmm is already equivalent
   for trees; also explicit `hour` categorical: ±0. Feature space is saturated.
4. **Bootstrap resampling per member** (0.7181 vs 0.7184), **feature-subset bagging**
   (0.7182), **lossguide members** (0.7182), **rank-average / logit-average / softmax-weighted
   aggregation** (all 0.7211 = no change): diversity via data/resampling adds nothing here;
   probability-mean over diverse configs is as good as it gets.
5. **dart members** and one-hot variant members: dart +0.0001 (kept, cheap); one-hot on
   ~700 dense columns × 100k rows blew the 120 s timeout (dropped).

## With more budget

I would (a) build a proper stacked ensemble: OOF member predictions on train (5-fold) feeding
a small meta-XGBoost, which needs ~10 min of compute per run; (b) tune member count vs.
runtime (early stopping accounts for most of the 92 s); (c) probe interactions of `doy` with
hour at *low* cardinality (e.g., season × daypart, 16 cells) since only year-stable features
transfer; (d) test quantile-bin DepTime and Distance per-route occupancy as drift-robust
numerics. The 2005→2006 drift gap (~0.03 AUC vs within-2005 validation) is the dominant
remaining error source; anything that estimates drift-robust conditional effects would pay.
