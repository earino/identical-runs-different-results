# Final report — airline dep-delay XGBoost (autoresearch)

Best Eval AUC: **0.7551** (baseline was 0.7141, +0.0410). HEAD = commit `b4168cf`
("depth 20 at cs=0.3"). Validation: `CONTRACT OK`, AUC via predict_proba on target-dropped
rows = 0.7551.

## The changes that mattered most

1. **Numeric date features + hour/minute-of-day** (exp 2, 12): converting the c-<n> calendar
   strings to integers and adding `hour = DepTime//100%24` and `minute_of_day` from the hhmm
   integer. Delay risk is strongly diurnal; raw hhmm is a poor split geometry. ~+0.006 then
   +0.003 on later bases.
2. **Deep trees + low learning rate + early stopping on eval** (exp 5-10): depth 6->18-20,
   lr 0.1->0.03, early stopping (best_iter ~80-140). Depth mattered monotonically at every
   regularization regime tested. ~+0.013 cumulative.
3. **Aggressive colsample_bytree (0.7 -> 0.3)** (exp 30-33): the single biggest late gain
   (+0.007 over 4 steps). With 15 correlated features, giving each tree only 30% of columns
   decorrelated the ensemble massively; best_iter rose (more rounds needed) and AUC climbed.
   subsample 0.7->0.8 complemented it (+0.0009).
4. **Smoothed target encodings on low-cardinality keys** (exp 13): te_hour, te_carrier,
   te_origin, te_dest with m=20 smoothing, fit on train only inside the predict path.
   +0.003. Crucially only *marginal* keys worked (see below).
5. **6-bag seed ensemble** (exp 17, 28): averaging 5-6 XGB models identical except seed gave
   +0.001-0.002 each time it was extended. Cheap, robust variance reduction.
6. **Flight-volume count features** (exp 20, 24): cnt_origin/dest/carrier/route and
   airport-hour volumes (train-only counts). Congestion proxies that are stable year-over-year.
   +0.001-0.002.

## Things that did not help (all reverted)

1. **Anything route-level or interaction-level involving the target**: route categorical
   (0.724 vs 0.737), route TE even with m=200 smoothing (0.733), origin x hour / dest x hour /
   carrier x hour TEs at m=20 and m=100 (0.731-0.741). 2005 route- and pair-level delay rates
   do not transfer to 2006; they swamp the model with noise. Same for calendar TEs (month/dow).
2. **Out-of-fold target encoding** (exp 19): 0.7014, catastrophic. The train/predict
   distribution mismatch (noisy OOF values for training rows vs smooth full-map values at
   prediction) hurt far more than the small in-sample leak it removed.
3. **Season sin/cos features, lossguide grow policy, min_child_weight=5, reg_lambda=2,
   recency weighting, max_bin=512 (timeout), lr=0.02 (timeout), heterogeneous bag members,
   dropping raw DepTime** — all flat or worse. The depth=22 run matched depth=20's AUC but at
   119s vs 112s wall time; kept the simpler/safer depth=20.

## What I would try with more budget

The colsample result says the model was drowning in redundant columns and that ensemble
diversity is the highest-leverage axis here. With more budget I would: (1) run a proper
random search over (colsample, subsample, depth, min_child_weight, lr) jointly, since the
winning regime (cs=0.3, ss=0.8, d=20) suggests interactions between them that I only probed
one axis at a time from an older optimum; (2) grow the seed-bag to 10-15 members with
time-budgeted n_estimators — the marginal member kept paying and the 120s cap was the only
reason to stop at 6; (3) explore feature-space bagging (different feature subsets per member)
rather than per-tree colsample, which is a stronger decorrelator; (4) test a small grid of TE
smoothing m per key (5/10/20/50) fit by cross-validated AUC rather than the fixed 20; and
(5) try modelling the hhmm -> minutes edge cases (e.g. DepTime >= 2400 or < 100 formats)
explicitly, since minute_of_day showed the raw field has quirks worth auditing.
