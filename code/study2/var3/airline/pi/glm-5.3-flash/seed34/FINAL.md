# Final Report — airline delay AUC

**Best Eval AUC: 0.7459** (baseline: 0.7141, +0.032)

Final model: 8-seed XGBoost bag (`colsample_bytree=0.35, max_depth=20, min_child_weight=1.0, n_estimators=68,
lr=0.05, subsample=0.85, max_bin=512, hist, native categoricals`), trained on all of train.csv with month-recency
sample weights (Dec = 3× Jan), predictions averaged arithmetically. Config was selected by an internal 2-fold CV
over a grid (train data only); n=68 came from early stopping inside that CV.

## Changes that mattered most

1. **Time parsing / basic FE** (exp 2, 0.7141 → 0.7208): numeric month/day-of-week/day-of-month, DepTime → hour +
   minute-of-day, log1p(Distance), distance buckets, native categoricals for carrier/origin/dest.
2. **Strong column subsampling + CV-based hyperparameter selection** (exp 10–13, → 0.7283): colsample_bytree 0.4–0.6
   was the dominant hyperparameter (0.8 → 0.6 alone was +0.007 CV); an internal 2-fold CV grid made selection honest
   and transferable to 2006.
3. **Flight-volume features** (exp 26–28, 0.7323 → 0.7357): log1p counts of flights per Origin, Dest, Route,
   Carrier, Carrier×Origin pair, plus origin/dest/carrier network degrees — target-free statistics that transfer
   across the 2005→2006 year shift (unlike target encodings, which hurt).
4. **Month-recency sample weighting** (exp 22–24, +0.006): upweighting late-2005 months (Dec = 3× Jan) aligns the
   fit with the 2006 target year; Dec 3× was the sweet spot (4× slightly worse).
5. **Deep-trees/few-rounds regime + seed bagging** (exp 34–38, 0.7361 → 0.7459): after the feature upgrades the CV
   optimum moved to depth 16–20 with only ~67 trees (ES), min_child_weight 1.0, col 0.35; an 8-seed bag of that
   config added +0.005 over a single model.

## Things that did not help

1. **Target encoding** (smoothed and 5-fold OOF, 9 key sets incl. route and hour interactions): 0.6658–0.7147 —
   the 2005→2006 shift makes train-fitted target statistics misleading; native categorical splits beat TE here.
2. **High-cardinality interaction keys as native categoricals** (Origin×Hour, Route, Carrier×Hour): 0.7173 — too
   sparse, overfits 2005 specifics. Small keys (Month×DOW, DOW×Hour) and calendar extras (DayOfYear, weekend,
   holiday windows, cyclical sin/cos) were all neutral-to-negative (0.7205–0.7333).
3. **Volume × evening products and mean-distance-per-hub features** (0.7337–0.7345): explicit interaction products
   and hub-profile stats added noise the trees could already capture; deeper/shallower single models and logit-space
   blending were also flat (0.7310–0.7323 vs 0.7310).

## With more budget

I would (a) run a larger 5-fold CV grid jointly over depth/min_child_weight/colsample/learning-rate in the
deep-trees regime (the optimum moved a lot when features changed, and 2-fold estimates are noisy), (b) explore
feature-set diversity in the bag (members trained on disjoint feature subsets), (c) probe sample-weight shapes
other than linear-in-month (e.g., day-of-year exponential decay) and per-member early stopping, and (d) revisit a
carefully regularized route-level encoding using only year-stable aggregates (e.g., encoding computed on
holiday-adjusted residuals).
