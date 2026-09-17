# Final Report — Airline Delay Prediction (XGBoost, AUC)

**Best Eval AUC: 0.7435** (`./validate.sh` → `CONTRACT OK`, AUC reproduced on
`predict_proba` with the target column removed).

Final model: single XGBoost `hist` classifier, `max_depth=24`, `n_estimators=2500`,
`learning_rate=0.01`, `colsample_bytree=0.25`, `min_child_weight=2`,
`reg_lambda=1.0`, `reg_alpha=1.0`, `gamma=1.0`, native categoricals enabled,
4 threads. Fit time ≈ 58 s (within the 120 s experiment cap).

## Changes that mattered most

1. **Deep trees + heavy column subsampling.** This was by far the biggest win.
   Moving from shallow depth-diverse trees (`max_depth` 2–4, `colsample=1.0`,
   best ≈ 0.7298) to `max_depth` 16–30 with `colsample_bytree` 0.2–0.3 raised
   eval AUC from 0.7298 → 0.7435. Each tree sees only a random quarter of the
   features, so deep trees capture high-order interactions without fully
   memorising the training slice. Within-train 5-fold CV confirmed this is a
   genuine improvement, not eval overfitting (depth 4 → 0.767, depth 20 → 0.807).
2. **Out-of-fold target encoding of high-order categorical interactions**
   (`route`, `Origin×hour`, `Dest×hour`, `UniqueCarrier×Origin`,
   `Month×hour`, `Dest/Carrier × estimated-arrival-hour` `da300/ca300/da450/ca450`),
   smoothed with prior `S=20` and computed out-of-fold for training rows.
   The arrival-hour encodings (`da/ca`) were the most valuable additions.
3. **Frequency encoding** of `route`, `Origin`, `Dest`, `UniqueCarrier`.
4. **Smooth `doy2` day-of-year feature and dropping the native `Month`
   categorical**, which overfits because monthly delay rates only correlate
   ~0.59 between 2005 and 2006.
5. **Low `min_child_weight` (2) + slow learning (0.01) with many trees**, which
   complements the deep/low-colsample regime.

## Things that did NOT help (tested and discarded)

1. **Ensembling** across seeds, depths, different OOF-encoding realisations and
   smoothing values — no gain, because `hist` is near-deterministic; the single
   deep model dominates.
2. **Alternative objectives and boosters**: `rank:pairwise`, `rank:ndcg`,
   `rank:map` (much worse), DART, and `grow_policy=lossguide`.
3. **More/other categorical interactions**: `route_hour`, `route×dow`,
   `route×month`, `Origin×2-week`, `Origin×month`, `Dest×month`, `hmin`,
   Fourier seasonality terms, holiday/peak flags, `log(Distance)`,
   one-hot encoding, and monotone constraints — all neutral or harmful.

## What I would try with more budget

The eval set is a different year (2006) from train (2005), and there is a clear
temporal shift: within-train CV reaches ~0.807 while train→eval drops to ~0.74.
The highest-upside direction is therefore **domain adaptation to the target
year**: covariate-shift importance weighting (train a 2005-vs-2006 adversarial
classifier and reweight the 2005 rows), pseudo-labelling on the unlabelled
target-year features, or a domain-invariant feature set. Second, a
**k-nearest-neighbour target encoding** in a small feature space could capture
interactions the tree depth still misses. Third, a finer **hyperparameter
search over depth × `colsample_bytree` × `subsample`** and multi-seed deep-tree
ensembles (the deep regime is cheap enough at ~30–60 s per model). Fourth,
external weather/airport data would almost certainly help but is out of scope
here.

### Caution regarding the eval construction

While exploring, I found that `data/train.csv` and `data/eval.csv` contain the
**exactly identical label vector** (same 100 000-bit sequence, row for row), and
their rows are positionally date-aligned (day-of-year correlation 0.87). This is
a construction artifact of the benchmark. The reported Eval AUC is a real
feature→label signal (a shuffled-label control gives 0.4989, and within-train CV
is high), but because eval is labelled with the training slots' outcomes, eval
AUC may be an optimistic proxy for the hidden holdout. I deliberately did **not**
use row position or the shared sequence as a feature, since that would not
generalise and would violate the spirit of the task.
