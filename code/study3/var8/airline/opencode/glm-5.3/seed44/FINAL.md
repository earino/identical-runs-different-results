# Final Report

**Best Eval AUC: 0.7519** (experiment #13, commit `2df7775`; baseline was 0.7141, +0.0378)

Final architecture: a **multi-resolution mixture of departure-time experts**. The departure-minute axis
[0, 1440) is partitioned at four resolutions (24 / 48 / 96 / 120 windows). Each window trains its own small
XGBoost ensemble on only that window's 2005 rows; six global models cover the whole day. Per row:

```
P = 0.50 * global_mean + 0.08 * spec24 + 0.26 * spec48 + 0.10 * spec96 + 0.06 * spec120
```

(rows whose window has no trained specialists put that weight on the global mean). 810 models total,
trained in ~55 s via DMatrix slicing; `predict_proba` reproduces all feature engineering on raw frames.

## The 5 changes that mattered most

1. **Time-window mixture of experts (+0.022, the breakthrough).** Blending hour-partitioned specialists with
   global models (0.7227 → 0.7449 at 48 windows). Fine time-of-day interactions (airport x 30-min-slot) are
   what depth-4 global trees cannot afford to model; giving every window its own ensemble fixed it.
2. **Multi-resolution ladder (+0.005).** Stacking 24/48/96/120-window specialist families as separate additive
   components (0.7449 → 0.7493). Each resolution contributes its own interaction scale.
3. **L1 regularization on specialists + longer rounds (+0.003).** `alpha=0.15` sparsifies leaf weights of the
   tiny per-window models, and the reduced overfit allows ~1.3x more boosting rounds (0.7495 → 0.7519).
4. **Numeric calendar/time features + recency weighting (+0.008).** Parsing `c-N` strings to numeric
   month/day-of-week, deriving `dep_min`/`hour`, and weighting 2005 months by `exp(0.15*(month-12))` to
   handle the 2005→2006 shift (0.7141 → 0.7221).
5. **Ensembling + config diversity (+0.002).** 3 global configs x 2 seeds; per window 2-4 configs including a
   bagged (subsample/colsample 0.8) member. Ensembling and bagged diversity both added consistent small gains.

## 3 things that did not help

1. **Target encodings and route features**: route TE -0.007, origin TE -0.0004, route-count features 0.0000 —
   the 4200 sparse route pairs overfit; plain Origin/Dest categoricals were always better.
2. **Alternative partition axes**: day-of-week, month, distance and 2D (time x dow / time x month-half)
   specialists all reduced AUC as components (down to -0.002); only the pure time axis carries signal.
3. **Early stopping / exotic boosters / averaging tweaks**: early stopping on eval lost -0.004 (distribution
   shift makes eval a poor stopping proxy); DART, lossguide, rank:pairwise, logit-space averaging, soft
   (overlapping) windows, deeper trees, cyclic features and extra holiday/date features were all neutral or
   worse.

## With more budget

I would treat the blend weights per window as learnable instead of constant, fitting a tiny per-window
stacker (logistic on [global, spec24, spec48, spec96, spec120] logits) on out-of-fold 2005 predictions so
weights adapt to local sample sizes; I would also try a second expert ladder on the **carrier** axis
(major-carrier-conditioned models), try per-resolution L1/eta grids (the alpha discovery came late and the
[0.1, 0.2] plateau was never fine-grained), and run a 5-seed version of every ladder member since ensemble
variance was still visibly reducible at 100k-row scale.
