# Final report — airline dep_delayed_15min, XGBoost

## Result

- Baseline (30 trees, raw columns): **Eval AUC 0.7141**
- Final: **Eval AUC 0.7532** (+0.0391) — commit `3daa6f5`, validated (`CONTRACT OK`, predict_proba path reproduces 0.7532).

15 experiments run (of 40); CPU compute budget (18,000 CPU-s) was the binding constraint, not the experiment count.

## The 5 changes that mattered most

1. **Numeric time-of-day decomposition** (Hour, Minute, DepMin from hhmm DepTime; Month/Day/Dow parsed from `c-<n>` strings). DepTime hhmm as a raw integer forces the trees to split inside the "minute" digits, which is meaningless; Hour/DepMin give clean congestion splits. (+0.005 alone, prerequisite for everything else.)
2. **Airport/carrier x hour-of-day interaction categoricals** (OriginHour, CarrierHour, DestHour, OriginHalf, DestHalf). Delay risk is a function of scheduled time-of-day at a specific airport (banks, queues, curfews); a tree cannot express "ORD at 17:00" from separate features cheaply. This was the single biggest feature-engineering win (+0.007).
3. **Recency-weighted training**: sample weight x2 for the last three months of the training year (train=2005, eval/holdout=2006). Per-month delay rates drift year over year; upweighting the temporally closest data cut the drift penalty. (+0.002, and dropping it later cost exactly that.)
4. **Strong L1 regularization (reg_alpha) unlocking depth**: alpha 4-8 at depth 7-8 beat alpha 0 at depth 4. L1 prunes the noisy high-cardinality splits that depth would otherwise exploit. (+0.006 over the unregularized optimum at the time.)
5. **Feature-bagged ensemble**: 2 full-feature members (d8, alpha=2) + 8 members each seeing the core numeric/time/volume features plus a random 4 of the 8 categorical/interaction features, averaged. Feature-subset diversity decorrelates members far better than seeds or hyperparameter jitter; +0.006 over the best single model. Airport-hour **volume features** (train-set flight counts per Origin/Dest x Hour, raw and airport-normalized) as always-on core features added +0.002 inside the ensemble.

## 3 things that did not help (tried, measured, reverted)

1. **Target encoding** of any column (route, origin, hour, month): fit on 2005, harmful on 2006 — the per-level delay rates are not stationary across years, so the encoder just injects 2005 noise (-0.015).
2. **Row subsampling / bootstrap of ensemble members**: every member loses effective rows and the mean gets worse (-0.002 vs feature-bag-only). Member diversity must come from features, not rows, at this data size.
3. **Higher-cardinality interactions** (Route origin_dest 29k levels, Carrier x Dow x Hour, MonthHour, day-of-year/week-of-year calendar features): all neutral to negative; the two-way airport x hour interactions already capture the transferable structure, and calendar specifics of 2005 do not transfer.

## What I would try with more budget

The model is time-separated-constrained: with no arrival times or same-day earlier-flight outcomes, delay propagation is invisible, and the afternoon block (12:00-18:00, where AUC is worst at ~0.66) is where the missing signal lives. With more budget I would (a) build explicit "scheduled-bank density" features — counts of same-carrier/same-airport departures in a +-1h window around each flight, requiring only a groupby, which is a richer congestion signature than the single-hour counts I used; (b) tune per-segment: separate members (or monotone constraints) for the afternoon block where the average effect is learned badly; (c) replace the fixed 4-of-8 feature bag with a Bayesian/successive-halving search over member feature sets, since member composition moved the needle most; and (d) verify every candidate on a 2005->2005 time split inside train to stop overfitting keep/discard decisions to eval.csv, which remains the biggest methodological risk in this loop.
