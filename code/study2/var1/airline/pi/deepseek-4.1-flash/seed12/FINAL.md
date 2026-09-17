# Final report — airline delay prediction (XGBoost)

**Best Eval AUC: 0.7263** (experiment #40, commit `dcc0991`)
Baseline: 0.7141. Total gain: **+0.0122**. Net gain over the best single model (0.7192) comes from ensembling.

The final `train.py` defines `predict_proba(df)` and passes `./validate.sh` (`CONTRACT OK`, eval AUC 0.7263
with the target column removed).

## What mattered most

1. **A small, well-regularized model beats a big one** (2005→2006 is a temporal shift). The baseline's
   `max_depth=6, n_estimators=30` overfit relative to the eval year. Lowering to `max_depth=4` and using
   `n_estimators≈150–500, lr=0.05` raised AUC from 0.7141 → 0.7191. More capacity (depth 5–7 at high tree
   counts, depth 7 with 400 trees, DART, `max_cat_threshold=256`) consistently hurt.
2. **Calendar features: replace raw month/day categoricals with a smooth day-of-year.** Parsing
   `Month`/`DayofMonth`/`DayOfWeek` from `c-<n>` and adding `doy=(month-1)*31+dom` plus `is_weekend`
   gave 0.7208 → 0.7225. Then **removing the `Month`/`DayofMonth` categoricals** (keeping `doy`) gave the
   single biggest jump, 0.7225 → 0.7254. The 12/31-level categoricals were fitting 2005-specific
   calendar noise; the ordered `doy` transfers across years.
3. **Time-of-day engineering.** `hour`, `minute`, `tod = hour*60+minute` from the raw integer `DepTime`
   (hhmm) added +0.0007 and made `DepTime` redundant (dropping it was neutral).
4. **Diverse-depth, bagged XGBoost ensemble.** Averaging models with different depths / learning rates /
   subsampling lifted AUC further: 5-member ensemble 0.7208 → 0.7254, and the final 8-member bagged
   ensemble (four depth-6 + two depth-7 models with `subsample=0.7, colsample_bytree=0.7`, plus one
   depth-4 `lr=0.03` and one depth-8 model) reached 0.7263. Seed-only averaging (identical configs) gave
   nothing; *config* diversity did.

## What did not help

1. **High-cardinality categorical interactions** — a `route` (Origin_Dest) categorical dropped AUC to
   0.7045, and `max_cat_threshold=256` to 0.7163. Finer categorical splits overfit 2005.
2. **Target/frequency encoding** — out-of-fold target encoding of carrier/origin/dest (0.7188 vs 0.7191,
   and 0.7261 vs 0.7261) and count/frequency encodings (0.7180, equal) were neutral; XGBoost's native
   categorical handling already captures the same signal.
3. **Extra time/feature variants** — cyclical `sin/cos` of `doy`/`tod` (0.7217), dropping `minute`
   (0.7230), dropping the `DayOfWeek` categorical (0.7215), dropping `Origin`/`Dest` (0.7065), DART
   booster (0.7139), and `subsample/colsample=0.8` on the single model (0.7171).

## With more budget

The score is dominated by generalization across the 2005→2006 shift, so I would invest in (a) *semi-
supervised / domain-adaptation* ideas that use the unlabeled eval-year feature distribution (e.g. feature
distribution matching or self-training with a confidence threshold), while still never touching eval
labels; (b) a proper out-of-fold target/route encoding with heavy smoothing implemented so it is identical
at inference; (c) a broader bagged ensemble (10–20 configs) with weights chosen on a *time-ordered*
validation split carved from train rather than on eval, to reduce selection overfitting on the 100k eval
slice; and (d) monotonic/regularization constraints on `doy` and `tod` to further smooth the seasonal and
diurnal effects that proved to be the most transferable signals.
