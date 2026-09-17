# FINAL — airline delay XGBoost (autoresearch)

**Best Eval AUC: 0.7264** (commit `6798de1`, experiment #36). Baseline was 0.7141.

## What mattered most

1. **Heavy L1 + L2 regularization.** `reg_alpha=10`, `reg_lambda=20`, `min_child_weight=10`.
   This was by far the largest gain (≈0.7141 → 0.7249 on a single model). The 2005→2006 temporal
   shift makes the model overfit very fast; shrinking leaf weights is what makes extra capacity usable.
2. **Time-of-day features from `DepTime`.** `hour`, `minute`, plus a *shifted-day* ordinal
   `t = ((hour-4) % 24)*60 + minute` (and its sin/cos). Delay risk is essentially monotone in time
   once the day starts at 04:00 (near-zero delay in the early morning, climbing to ~0.8 by late
   evening), so this ordinal is a much cleaner representation than raw `hhmm`.
3. **Low learning rate with many trees.** `n_estimators=1200`, `learning_rate=0.02`, depth 6.
   With the regularization above this beat both fewer trees and deeper/boosted variants.
4. **Native categorical handling** of `Month`, `DayofMonth`, `DayOfWeek`, `UniqueCarrier`, `Origin`,
   `Dest` (`enable_categorical=True`), with the category levels fixed from the training set so
   unseen levels become NaN rather than shifting codes.
5. **4-model XGBoost ensemble** (two depth-6 regularized seeds + one depth-7 and one depth-5
   regularized member), averaging predicted probabilities: 0.7249 → 0.7264. All diversity is
   XGBoost-only, as required.

## What did not help (and was reverted)

- **`route = Origin|Dest` as a categorical**: caused a large drop (0.726 → ~0.710). Route-specific
  2005 patterns do not transfer to 2006; this was the clearest overfitting signal in the task.
- **Frequency encodings and out-of-fold target encodings** of carrier/origin/dest/route: neutral to
  slightly negative (~0.7248–0.7249).
- **Extra calendar features** (day-of-year sin/cos), **more capacity** (depth 7/8, 1500–2000 trees),
  **DART** (exceeded the 120 s limit), **monotone constraints**, **rank-averaging**, and **dropping
  raw `DepTime`**: all neutral or worse.

## With more budget

I would treat the 2005→2006 shift as the central problem rather than tuning for eval AUC. Concretely:
build a time-ordered internal validation split inside `train.csv` (and/or an adversarial
train-vs-eval classifier) to select features and early-stopping rounds without touching `eval.csv`,
which is currently the only feedback signal and therefore easy to overfit after ~40 decisions.
On the feature side I would try airport/carrier aggregate statistics computed *within* time windows,
finer-grained origin×time-of-day intensities with strong shrinkage, and a two-stage model that first
predicts a baseline (hour/route) propensity and then a residual model, so the regularization is
applied where the signal actually lives.
