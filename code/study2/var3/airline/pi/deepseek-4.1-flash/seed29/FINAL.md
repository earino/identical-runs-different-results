# FINAL — autoresearch XGBoost (airline)

**Best Eval AUC: 0.7233** (commit `72501d1`, 10-fold out-of-fold target encoding + arrival-time features,
depth-4 XGBoost, 800 trees, lr 0.03, min_child_weight 20). Baseline was 0.7141.

## Changes that mattered most
1. **Time-of-day decomposition of `DepTime`** — `hour`, `minute`, and cyclic `sin/cos` of minutes-since-
   midnight (+0.0033 over baseline). Raw `DepTime` alone was the only time signal the baseline had.
2. **Estimated arrival time** — `block_min = 30 + 0.12*Distance`, `arr_tod = tod + block_min` and its
   sine/cosine, plus `arr_hour` proxies. Delays accumulate over a flight's day, and arrival time captures
   this better than departure time (+0.0017).
3. **Out-of-fold smoothed target encoding** (10 folds, alpha=100) for `UniqueCarrier`, `Origin`, `Dest` and
   their hour-bucket interactions `origin_hr`, `dest_hr`, `carrier_hr`. A separate model is trained per fold
   so no training row sees its own target in its features; predictions are averaged over folds (+0.0015
   combined vs. non-OOF, and markedly more robust).
4. **Shallow, strongly regularized trees** — `max_depth=4`, `min_child_weight=20`, `lr=0.03`, 800 trees
   (+0.0017 vs. the 30-tree depth-6 baseline). The 2005→2006 time shift punishes capacity.
5. **Keeping raw `Origin`/`Dest` categoricals** alongside their target encodings (removing them cost 0.0044).

## Things that did not help
1. **High-cardinality memorization**: `route` categorical (−0.013), `route` target encoding (−0.010 even
   with OOF), route frequency encoding (neutral). Route/temporal latents do not transfer across years.
2. **Calendar interactions**: `origin/dest × DayOfWeek` (−0.007) and `origin/dest/carrier × Month` (−0.014)
   target encodings; weekly/monthly airport latents are mostly year-specific noise.
3. **More/sampled complexity**: 400–1500 trees at low lr, `depth=5`, `subsample`/`colsample=0.8`, L1/L2
   penalties, seed ensembling and 2-seed-per-fold averaging all tied or lost. Extra stochasticity and
   capacity only added variance.

## With more budget
The clear remaining direction is better handling of the temporal distribution shift rather than more
feature memorization. A weighted/recency scheme is unavailable (train is a single 2005 slice), but one could
train a small set of XGBoost models on disjoint temporal/route strata and blend them by estimated year
stability, use adversarial validation to drop features whose train/eval distributions diverge most, and
replace fixed smoothing with a per-key empirical-Bayes prior. A monotone constraint on the departure/arrival
time features, and calibration-free rank averaging over several OOF fold counts (5/10/20), are cheap follow-ups
likely worth ~0.001–0.002.
