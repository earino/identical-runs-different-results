# FINAL — airline delay 15+ (XGBoost AUC)

**Best Eval AUC: 0.7485** (16-member XGBoost ensemble, commit `66b01be`), up from the 0.7141 baseline (+0.034).
Validation: `./validate.sh` → `CONTRACT OK`, `predict_proba` reproduces 0.7485 on a target-free DataFrame.

## Final model

Uniform average of 16 `xgboost.XGBClassifier`s, all at `learning_rate=0.025, n_estimators=2000, colsample_bytree=0.7`,
tree_method="hist", trained on the same OOF target encodings but over different **feature views** and **depths**:
- views: `full` (raw cats + numerics + TEs), `nocat` (TEs + numerics, no raw cats), `teonly` (TEs + hour features only),
  `catsonly` (raw cats + numerics, no TEs — native categorical splits);
- depth ladder per view: d3, d4, d5 (+ d6 for the views where it helped; one `catsonly` d6/lr0.1/r200).

Target encodings (smoothed means, all fit on train only; train rows get 5-fold out-of-fold values):
`hour×origin/dest/carrier`, `dow×origin`, `hour×route`, and hierarchical flight-fingerprint TEs with count
shrinkage m=5 toward a coarser prior (m=25): `carrier|route|DepTime → route`, `origin|DepTime → origin`,
`dest|DepTime → dest`, `carrier|route|hour → carrier|route`. Raw `hour/minute/tod` numerics.

## Changes that mattered most

1. **Schedule-fingerprint target encodings** (+0.02 cumulative): `te_hour_route` (+0.008) and the hierarchical
   flight-level TEs `carrier|route|DepTime`, `origin|DepTime`, `dest|DepTime`, `carrier|route|hour` (+0.015 together).
   Exact-DepTime keys beat hour-binned keys for origin/dest; hierarchical priors beat plain m-shrinkage.
2. **View-diversity ensembling** (+0.007): averaging models over different feature views (`full`, `nocat`, `teonly`,
   `catsonly`) far outperformed any single model. Adding the raw-only `catsonly` view was the single biggest jump
   (+0.003), despite being the weakest member alone — its errors are decorrelated from the TE-heavy views.
3. **Depth ladder inside the ensemble** (+0.001): each view at d3/d4/d5/d6 beats any single depth; deep members
   that overfit alone still help in the average.
4. **Shallow main model + many low-lr rounds** (0.05×400 → 0.025×2000, +0.0017): robust under the 2005→2006 shift,
   unlike deep single models (d6+ single models lost 0.01+).
5. **Out-of-fold TEs on train rows** as regularization (single-model +0.004 over leaky full-fit TEs; noisy OOF
   acts like shrinkage toward the prior, which generalizes better to 2006).

## Things that did not help

1. **Deep single models / capacity increases**: any single d6–d8 model lost 0.005–0.02 AUC on eval (2005→2006
   shift); early stopping on a 2005 holdout picked badly overfit round counts.
2. **Raw route as a categorical in a single model** (−0.015) and simple single-column TEs (carrier/origin/route alone,
   −0.003): the route effect is only usable through its interaction with time, and only via TE smoothing.
3. **Ensemble mechanics**: member weighting, rank-averaging instead of prob-averaging, seed doubling (same view,
   different seeds), subsample bagging, DART/lossguide members, spec-subset members, 10-fold OOF — all neutral or
   worse. Uniform probability averaging of view+depth variants was consistently at or near the top.

## With more budget

I would push the two axes that still showed slope: (a) more view definitions — e.g. views that drop individual TE
families (flight-only, inter-only) as ensemble members showed small but consistent gains when tried singly, and a
systematic "leave-one-feature-family-out" ladder could add a few more decorrelated members; (b) per-view round/depth
budgets (the shallow views may want more rounds). Beyond that I would attack the distribution shift directly:
domain-adaptive TE shrinkage (larger m for groups whose 2005 rate is unstable), time-decay weighting inside 2005 if
finer timestamps existed, and a light wrapper learned on OOF predictions (constrained logistic) trained to
non-negatively combine the 16 members — attempted manually with fixed weights it failed, but a *learned*, regularized
combination fit on OOF predictions might beat uniform averaging. Finally, feature-level: remaining unexplored
interactions are seasonal (month×route congestion) and distance×time-of-day (red-eye long-haul), which the depth
ladder can pick up cheaply if added as extra TE columns.
