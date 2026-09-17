# FINAL — autoresearch XGBoost (airline)

Best Eval AUC: **0.7589** (experiment #16, commit `1687240`; baseline was 0.7141 → +0.0448).
Validation: `./validate.sh` → `CONTRACT OK` (predict_proba reproduces 0.7589 on eval.csv with the
target column removed).

## Final model

5-member XGBoost ensemble over one feature view (`hist`, `enable_categorical`, sorted columns):

- full view, alpha=20
- airport view (Month/DayofMonth/DayOfWeek dropped), alpha=0.3
- airport view, alpha=1, depth=10, mcw=30
- month+dom view (DayOfWeek kept), alpha=3
- full view, alpha=20, lr=0.02 x 1800 trees

Shared member recipe: 1200 trees, lr 0.03, depth 8, mcw 20, colsample 0.7.

## Changes that mattered most

1. **Numeric date parsing + hour decomposition** (0.7141 → 0.7219): Month/DayofMonth/DayOfWeek parsed
   from `c-<n>` strings to integers, DepTime split into hour/minute. The baseline's categorical
   treatment of time variables was leaving signal on the table.
2. **Entity x hour interaction categoricals** (0.7219 → 0.7322): crossing
   UniqueCarrier/Origin/Dest/DayOfWeek with departure hour as categorical features was the single
   biggest feature win; delay rates by hour differ strongly per carrier and per airport.
3. **Slow deep trees** (0.7322 → 0.7352): 1200 trees, lr 0.03, depth 8, mcw 20 beat both shallow
   and deeper/shallower mixes; the 2005→2006 shift punishes aggressive fast-learning trees.
4. **Strong L1 regularization + regularized members blend** (0.7352 → 0.7456): reg_alpha alone
   1-3 gave +0.008 solo; blending members spread across the alpha spectrum (0.3…15) decorrelates
   them and beat blending strong-member-only sets.
5. **Feature-view diversity** (0.7456 → 0.7589): members trained without the date columns
   ("airport view") are individually weaker but nearly independent; averaging full-view and
   date-less members added +0.013 over the best same-view blend. Final +0.0003 from a
   day-of-week-kept view and a slow lr member.

## What did not help

- **Route (Origin_Dest) as a categorical** — hurt in every config (-0.005 to -0.010); too sparse,
  and the Origin×hour/Dest×hour interactions already carry the route signal.
- **Target / frequency encodings, OOF stacking, weight optimization** — all ≤ +0.0002 or negative;
  the XGB meta-learner overfit the OOF fold structure (0.8278 OOF → 0.7564 eval vs 0.7584 mean).
- **Month×DayofMonth, Month×hour, route×hour interactions** — all reduced AUC (over-parameterized
  interactions on 100k rows under distribution shift).

## With more budget

I would (a) run a wider regularized-member grid search (alpha×gamma×colsample×mcw) with 3 seeds per
config and select on train-CV to make blend decisions shift-robust, (b) explore time-decay sample
weighting inside 2005 (recency weighting by month) instead of train-vs-eval reweighting, which
failed here, and (c) test per-carrier hour interaction targets (carrier×origin×hour) at a coarser
smoothing than the plain interactions that already worked.
