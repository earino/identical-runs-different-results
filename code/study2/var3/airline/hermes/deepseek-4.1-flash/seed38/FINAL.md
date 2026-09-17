# FINAL — airline delay classifier (XGBoost)

**Best Eval AUC: 0.7411** (experiment #40, commit `c9beb93`), reproduced by `./validate.sh` ->
`[validate] eval AUC via predict_proba (target column removed): 0.7411`, `CONTRACT OK`.
Baseline was 0.7141, so the loop gained +0.0270. All 40 experiments were used; no crashes in the
final config (single run ~95s, inside the 120s limit; 8,982 of the 18,000 CPU-seconds).

## Changes that mattered most

1. **Time-of-day features + ordinal decoding of the `c-<n>` columns.** `DepTime` (hhmm as an integer)
   became `dep_hour`, `dep_min`, minutes-since-midnight, and cyclic `tod_sin`/`tod_cos`; `Month`,
   `DayofMonth`, `DayOfWeek` were decoded to integers instead of being fed as categoricals.
   (`0.7141 -> 0.7154`; an ablation that removed the cyclic time transforms later cost a full `-0.009`,
   so the representation of time is where most of the signal lives.)
2. **Boosting-round selection with a shrink factor.** The round count is chosen by early stopping on an
   internal 15% split of train, then multiplied down: `1.0 -> 0.85 -> 0.70 -> 0.55 -> 0.40 -> 0.30`
   gave `0.7327, 0.7338, 0.7346, 0.7351, 0.7354, 0.7356` and 0.20 fell back to 0.7353. The round count
   that maximises 2005 validation AUC systematically overshoots the 2006 optimum, so truncating at ~30%
   of it (351-394 rounds at eta=0.02) is a cheap, principled fix for the train/eval year gap.
3. **Capacity + regularization regime.** depth 12, eta 0.02, subsample 0.6, colsample_bytree 0.6,
   lambda 5, `max_cat_threshold=256`, and the low-level `xgb.train` API (XGBoost 3.4's sklearn API no
   longer supports early stopping). This took 0.7226 -> 0.7327.
4. **`min_child_weight` driven to the default 1**, once rounds were truncated: 20 -> 10 -> 5 -> 2 -> 1
   gave `0.7357, 0.7384, 0.7398, 0.7406, 0.7411`. With few rounds the split constraint was the binding
   regularizer; early stopping had already replaced it.
5. **6-seed bag with probability averaging** (bag size was cheap after truncation; `+0.0001` only —
   minor, kept because it is free now).

## What did not help

1. **Target-rate and count encodings** for carrier/origin/dest/route, fit out-of-fold on train: 0.7149
   vs 0.7154 baseline-adjacent, and congestion counts were exactly neutral (0.7226 vs 0.7226). XGBoost's
   native categorical splits already capture the same information, and 2005 rates do not transfer to 2006.
2. **Route (`Origin_Dest`) as a categorical**: 0.7100, by far the worst run — 4,198 levels let the model
   memorise 2005 route quirks (385 eval routes are unseen anyway).
3. **Extra calendar structure** (month/day-of-week sin-cos + weekend flag, 0.7306), **lossguide growth**
   (equal but ~60% slower), **depth 16** (0.7325), **`max_cat_threshold=512`** (0.7279), **L1 `reg_alpha`**
   (drove the probe to 2,303 rounds and timed out), **two-config deep+shallow ensembles** (0.7320),
   **per-member sampling diversity** (0.7323), and **round counts averaged over two validation splits**
   (0.7317) — all reversible, all reverted.

## With more budget

The clearest next lever is making the stopping-round estimate itself robust: the single 15% split is
noisy (probe optima ranged 1,038-1,459 across near-identical configs), so k-fold averaged round
selection at the truncated point, plus per-seed round variation, should beat a single shared count.
Beyond that I would re-test the feature families that failed *at the final early-stopped regime* rather
than at the over-long regime where they were measured (min_child_weight told exactly this story: 20 was
best when the probe ran long, 1 is best now), try origin-hour congestion bands at finer granularity,
add monotone constraints on `tod`, and weight training rows to match eval's monthly mix. A larger bag
(10-15 members) is now nearly free at ~350 rounds and is the lowest-risk remaining gain. Validation
remains the weak point: every keep/discard decision here used 2006-slice1 only, so gains below ~0.001
should be treated as noise on the hidden holdout.
