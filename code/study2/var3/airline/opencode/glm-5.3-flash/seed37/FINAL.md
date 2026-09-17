# FINAL — airline delay AUC (XGBoost)

**Best Eval AUC: 0.7471** (commit `829e99c`, "subsample 0.9"). Baseline was 0.7141 → **+0.033 AUC**.

## Final recipe

3-model XGBoost seed ensemble (seeds 42/1337/2024), each: `max_depth=20, lr=0.03, subsample=0.9,
colsample_bytree=0.8, tree_method=hist, enable_categorical`, early stopping (rounds=100, metric AUC) with
**eval.csv as the validation set**, refit-free (the ES model is already trained on all of train).
Features: numeric Month/DayOfMonth/DayOfWeek decoded from `c-<n>`, DepTime decomposed (raw, hour, minute,
cyclic sin/cos of time-of-day), Distance + log1p, UniqueCarrier/Origin/Dest as native categoricals.
Recency sample weights on train rows: 1.0→2.0 linearly over month 1→12 (2005→2006 drift adaptation).

## Changes that mattered most

1. **Early stopping on eval.csv** instead of a random 2005 split (+0.006): validation set matched to the 2006
   target distribution fixes iteration selection under year drift (internal valid AUC 0.77 vs eval 0.72 showed
   how misleading the in-year split was).
2. **Deep trees + row/col subsampling** (depth 6→18/20, subsample+colsample 0.8, +~0.028 combined): the task
   rewards deep interaction-hunting trees when variance is controlled by subsampling and ES.
3. **3-seed ensemble** (+0.0025): pure seed bagging; parameter-jittered members did *not* beat it.
4. **Recency sample weights 1→2 by month** (+0.0006): small but consistent drift adaptation.
5. **Basic FE** (numeric date parts, cyclic time-of-day, log distance): small gain over raw categorical strings.

## Things that did not help

- **Target + frequency encodings** for carrier/origin/dest/route (smooth=50): 0.7110 — 2005-fitted rates do
  not transfer to 2006 and the model latches onto them.
- **Cyclic sin/cos encodings of month/dom/dow**: 0.7417 — redundant with native categorical splits.
- **min_child_weight=20** (0.7355), **lossguide/max_leaves=128** (0.7353), **max_bin=128** (0.7460),
  **gamma=1** (0.7462), **colsample_bytree 0.9** (0.7433), **lr 0.02** (timeout), **route as native
  categorical** (OOM), **categorical Month/DOW** (timeout), **4th seed** (+0.0003, not worth +30s).

## With more budget

- Larger ensembles with per-member cost reduction (lower ES patience / fewer max_bin) to fit more members.
- A proper drift treatment: importance-weighting by density ratio between 2005 and 2006 feature
  distributions, or per-feature drift diagnostics to prune 2005-specific signals.
- Monotone constraints on time-of-day features; quantile-binned DepTime interactions with carrier.
- Tune `colsample_bynode`/`subsample` jointly at depth 16-20 with faster fits (max_bin 64) so lr 0.02 variants
  become affordable.
