# Final report — airline departure-delay AUC

**Best Eval AUC: 0.7594** (experiment #40, commit `c3df6c8`), up from the 0.7141 baseline.
Budget: 40/40 experiments, ~3378/18000 CPU-seconds, ~24 min wall clock.

## Setup that won
A bagged ensemble of 18 XGBoost classifiers (depth 4–6, lr 0.03, varied
`colsample_bytree`/`subsample`/`min_child_weight` and seeds) averaged by probability, trained on
raw numerics + a large bank of **cross-fitted target-encoding (TE) features** and their
log-frequency counterparts. All feature engineering lives in `prepare(df)`, and the encoders are
fit on `data/train.csv` only, so `predict_proba` reproduces everything on unseen rows.

## Changes that mattered most
1. **Shallow, regularized trees instead of deep ones.** 500×depth-8 dropped the eval AUC to
   0.7032, and an internal-validation grid preferred deeper trees while *eval* consistently
   preferred depth 3–4. The 2005→2006 year gap means capacity overfits year-specific noise;
   depth 3–4 with many rounds was the robust optimum.
2. **Target encoding instead of native categorical splits.** Removing XGBoost's
   `enable_categorical` handling (which badly overfit `Origin`/`Dest`, and was catastrophic for
   route) and replacing it with smoothed TE features on categoricals gave +0.0026 immediately.
3. **Time-of-day interactions were the dominant lever.** Adding TE on `route×hour` (+0.008),
   then 30/15/10/5-minute bins (+0.010 more), then `carrier×route×time` (+0.005) drove the score
   from ~0.717 to ~0.758. Airport/carrier/route delay propensity is highly time-local.
4. **Cross-fitted (out-of-fold) encoding.** 5-fold OOF values for the training matrix avoid
   target leakage while inference uses the full-train mapping; this is what made the fine TE keys
   usable (10-fold was slightly worse than 5-fold).
5. **Ensembling shallow models.** Averaging 18 diverse shallow/medium XGBoost models gave a small
   but consistent +0.002–0.003 over the best single model.

## Things that did not help
1. **Deep, high-capacity models** (depth 6–8, 500–1200 trees): 0.703–0.713, clear year-shift
   overfitting despite better internal-validation AUC.
2. **Route as a native high-cardinality categorical** (0.7085) and **fine-minus-parent "excess"
   TE features** (0.7228): the latter amplified OOF-vs-full encoding mismatch.
3. **Month/seasonal interaction TEs and DayOfWeek×route interactions**: neutral to negative
   (0.7481–0.7489), likely because month-specific weather patterns don't transfer across years.

## What I would try with more budget
The feature engineering that paid off was all "delay propensity conditioned on where/when/who".
The natural next step is a **hierarchical (empirical-Bayes) encoding** where each fine key
(e.g. `carrier×route×5min`) is shrunk toward its parent (`carrier×route`, then `carrier`) rather
than toward the global prior — my flat smoothing may be discarding within-route signal in sparse
cells. I would also explore (a) an explicit **airport congestion proxy** built from train-set
flight volumes by origin/hour, (b) **rank-averaging** and weight-optimized blending of the
ensemble instead of a plain probability mean, and (c) a small hyperparameter search over
`reg_alpha`/`gamma`/`max_delta_step` with the now-rich feature bank. Finally, since selection was
done on a single 100k eval slice, I would use k-fold time-aware evaluation on 2005 to reduce
selection noise before committing to the hidden-holdout model.
