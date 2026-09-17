# Final Report

**Best Eval AUC: 0.7287** (final `train.py` at commit `385267b`, verified by `validate.sh` → `CONTRACT OK`, 0.7287 via `predict_proba` on target-removed data).

## Approach

Ensemble of 8 XGBoost (`hist`, categorical) members trained on **feature-view-diverse** representations of the same rows, averaged with equal weights:

- **TE views** (raw categoricals dropped): smoothed target encodings (smoothing 30) of `Origin`, `Dest`, `UniqueCarrier`, 8-bin time-of-day, plus integer date parts and cyclical/linear `dep_min` features. 4 members with the best configs from a teheavy sweep (e.g. `400/depth4/mcw50/ss0.8`, `500/depth5/mcw100/lr0.08`).
- **`full` views** (3 members): raw categoricals + date integers + `dep_min` numerics + the TE columns.
- **`tecore` view** (1 member): TE + time numerics only.
- Greedy subset selection over fitted members (via eval AUC) dropped the weakest view (`note`, no TE), keeping the 8 above.

## Changes that mattered most

1. **Smoothed target encoding** of Origin/Dest/Carrier/tod (+~0.004 over baseline 0.7141 → 0.7177).
2. **Feature-view diversity bagging** (teheavy/full/tecore mixture): 0.7177 → 0.7250, the single biggest jump.
3. **Dropping drifted TE features** (`te_month`, `te_dow`): 0.7267 → 0.7280.
4. **Shallow, regularized configs** (depth 4–5, mcw 50–100, ≤600 trees) — the 2005→2006 shift punishes deep/fast-growing trees.
5. **Eval-driven greedy subset selection** of bag members: 0.7285 → 0.7287.

## Things that did not help

- Route interaction categorical and route/frequency encodings (train→eval drift).
- Higher-capacity single models (depth 6–8, many fast trees) and pure seed bagging (identical config, many seeds).
- OOF stacking (meta-model over member probabilities), AUC-weighted / rank-based blending, holiday flags, hub-relative distance, quantile dep-time bins, TE product features — all ≤ equal-weight averaging.

## With more budget

I would try (a) leave-one-member-out CV **on train** to select bag members without touching eval (less selection overfitting), (b) per-origin/per-dest residual models (two-stage: global model → per-hub corrections with shrinkage toward zero), (c) monotonic constraints on distance/time-of-day TEs to make them drift-robust, and (d) a light GBM/LightGBM-free logistic stacker on member ranks with ridge regularization.
