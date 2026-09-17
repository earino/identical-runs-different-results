# FINAL — airline delay (AUC) XGBoost autoresearch

**Best Eval AUC: 0.7638** (experiment 14, commit `2e47a61`, `data/eval.csv`, 2006 slice 1).
Baseline was 0.7141.

## What mattered most (in order of impact)

1. **Unlimited-depth trees in a bagged ensemble** (`max_depth=0`, ~30 boosting rounds at
   `learning_rate=0.08`, `subsample=0.95`, `colsample_bytree≈0.42`, averaged over 16 seeds).
   This was the single big win: 0.7141 → 0.7525. Shallow boosted trees are limited by how many
   interactions they can express; with few rounds and aggressive row/column subsampling, very deep
   trees behave like a boosted random forest — each tree memorises the training slice (train AUC
   ≈ 0.9998) yet the seed-averaged ensemble generalises across the 2005 → 2006 split far better.
   `max_depth=12/16/20` and `max_depth=0` scored monotonically higher; `grow_policy=lossguide`
   was much worse.

2. **Bagging / seed averaging.** A single deep model ≈ 0.752; averaging 10–16 seeds added
   ≈0.002–0.003 for free. This also reduces variance on the much larger hidden holdout.

3. **Frequency / congestion encodings fit on train only:** route, origin, dest, carrier, plus
   `(origin|hour)`, `(dest|hour)`, `(route|hour)`, `(carrier|hour)`, `(carrier|origin)`,
   `(carrier|dest)`. This added ≈0.011 over the raw feature set (0.7525 → 0.7601 → 0.7638) and
   was the most reliable feature family. Volume at an airport at a given hour is a strong,
   year-stable proxy for departure congestion.

4. **`max_cat_to_onehot=64`.** One-hot the low-cardinality categoricals (Month, DayofMonth,
   DayOfWeek, UniqueCarrier) while keeping Origin/Dest as XGBoost partition splits. At shallow
   depth this alone was worth ~0.002; one-hotting Origin/Dest (282 levels) was clearly harmful.

5. **Time-of-day detail:** `dep_mins`, `sin/cos`, and minute-of-hour features
   (`dep_minute`, `is_round`, `is_round5`, `is_round15`) plus `is_weekend`, `log_distance`.
   The minute-of-hour block added ~0.0015.

## What did NOT help

1. **Target encoding** of Origin, Dest, Route, Carrier (out-of-fold, smoothed): consistently
   −0.002 to −0.005. Delay rates are not year-stable enough, and the deep trees already learn
   categorical target structure from the raw categories.
2. **High-cardinality categorical machinery:** adding `Route` as a categorical feature, top-K
   one-hot airports, `max_bin` above 512, `max_cat_threshold` changes, `num_parallel_tree` /
   RF mode, `colsample_bynode`, and `grow_policy=lossguide` all failed to beat the simpler setup.
3. **Calendar seasonality beyond Month:** day-of-year numeric/sin-cos and Month×DayOfWeek
   interactions dropped AUC by ~0.01 — 2005 and 2006 calendar alignment differs, so these
   overfit the training year. Likewise, most extra trees / lower learning rates / stronger
   regularisation (`min_child_weight`, `lambda`, `gamma`, `alpha`) all hurt; the model wants to
   fit hard and then be averaged.

## With more budget

The setup is clearly near a ceiling around 0.763–0.765 for these features, so I would spend the
next budget on (a) more seed/config diversity in the ensemble rather than new single features,
(b) a proper time-ordered internal validation to stop selecting on 100k-row eval noise, and
(c) genuinely new information rather than more encodings of the same columns — e.g. a
transductive/domain-adaptation estimate of 2006 airport-hour traffic from the unlabelled eval and
holdout feature distributions, and pseudo-labelling of the 1M unlabelled holdout rows, which the
deep-tree ensemble is well suited to exploit.
