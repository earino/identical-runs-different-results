# Final report — airline delay (XGBoost, hidden-holdout scenario 2)

**Best Eval AUC: 0.7489** (baseline 0.7141; +0.0348 over 40 experiments; contract-validated on `data/eval.csv` with the target column removed).

## Final model
Ensemble of **7 fixed-round XGBoost classifiers** (hist, `enable_categorical`), probability-averaged, each trained on the full 100k train rows with recency sample-weights (months of 2005 weighted linearly 1→2.5). Members vary in depth (16–30), colsample_bytree (0.4–0.6), learning rate (0.03–0.08), subsample (0.65–0.9) and seed; ~90–130 trees each; `max_bin=512`.

## Changes that mattered most
1. **Time-of-day feature engineering** (exp 2, 0.7141→0.7215): parsing `c-<n>` strings to ints, splitting DepTime into `hour`/`minute`, log-distance. Delay probability rises monotonically from 4% at 5am to 74% at 10pm — the dominant signal.
2. **Very deep trees** (exp 7–11, →0.7396): max_depth 8→24 gave +0.018. The data has strong high-order interactions (hour × carrier × origin) that deep depth-limited trees exploit; lossguide/min_child_weight/regularization variants all did worse.
3. **Early stopping + refit-on-full, then fixed rounds** (exp 6, 23): ES via an internal 80/20 split with **logloss** (not AUC!) and refitting on all data with the chosen round count was worth +0.0066 over using the 80%-fit models. Logloss-ES stopped at 72–150 rounds — early, which doubles as drift regularization toward 2006; AUC-ES trained 300+ rounds and scored worse on the time-separated eval (0.7411).
4. **Diverse low-colsample ensemble** (exp 14, 28–37, →0.7487): probability-averaging members with colsample_bytree 0.4–0.6 and depth 16–30 added ~+0.008 over the best single model; per-split diagnostics showed the low-col_sample deep members were individually strongest. Converting all members to fixed rounds (rounds taken from their ES runs) cut runtime 119s→85s with identical AUC.
5. **Recency sample weighting** (exp 24, 25, 40, →0.7489): weighting later-2005 months up to 2–2.5× added +0.001 — a cheap, consistent bridge across the 2005→2006 temporal split.

## Things that did not help
1. **Target encoding in any form** (exp 3, 4, 17): full-train smoothed TE (-0.005), Route-only TE (-0.005), and out-of-fold 5-fold TE (-0.007) all hurt. Year-over-year drift makes encoded rate features fragile; raw categorical splits are more robust.
2. **Extra redundant features** (exp 16): day-of-year and Origin/Dest frequency dropped AUC -0.004 — deep trees already combine primitives, extra columns just dilute colsample.
3. **Regularization of deep trees** (exp 13): min_child_weight=20 at depth 24 cost -0.014 (tiny leaves are load-bearing); lossguide/max_leaves=1024 (exp 12) and colsample_bynode=0.6 (exp 33) also lost badly.
4. **Ensemble saturation**: the 8th member (exp 38) and max_bin=768 (exp 39) were flat-to-negative.

## With more budget
First, a proper time-based internal validation (e.g., train on months 1–9 of 2005, validate on 10–12) to select rounds and weights against *temporal* rather than random splits, since that is where the generalization gap lives. Second, a randomized search over the member pool (depth 16–32, cs 0.3–0.7, lr 0.03–0.1) with greedy forward selection of ~10 members, plus a weighting scheme tuned on that temporal split. Third, origin×hour and carrier×hour interaction features tested with the ensemble (skipped here for runtime), and bagged recency-weight variants averaged to reduce sensitivity to the weight ramp.
