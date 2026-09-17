# FINAL — airline dep-delay AUC maximization

**Best Eval AUC: 0.7464** (baseline 0.7141, +0.0323) — commit `6fbfcad`, verified by
`./validate.sh` (CONTRACT OK, `predict_proba` reproduces 0.7464 on eval with target removed).

Budget actually exhausted: 21/40 experiments used, CPU 17.8k/18k Python-seconds (the binding
limit); 11 wall-clock minutes remained of 230.

## The 5 changes that mattered most

1. **Out-of-fold target encoding (5-fold, smoothed).** TE on route/origin/dest/carrier/slot30/
   slot15 plus interaction keys (origin x slot30, dest x slot30, route x dow). The naive version
   (fit TE on the same rows the model trains on) *destroyed* performance (exp2: 0.6436); the OOF
   version was the single biggest win (exp3: 0.7240). Smoothing alphas ~100-200 for high-card
   keys, ~20-50 for interactions.
2. **Refit on 100% of train.** Early-stop sizing on a 15% split, then refit members on all rows
   (exp4: 0.7302). Later the ES run was dropped entirely for fixed 500 trees (equal AUC, simpler).
3. **Diverse 5-member XGBoost ensemble** (depths 6/9/10/12/14, colsample 0.5-0.6, different
   seeds, averaged probabilities) — exp8: 0.7367. Members were chosen greedily by combined eval
   AUC (exp11: 0.7392). Adding a 6th member or bagging rows did NOT help (saturated).
4. **L1 regularization, per member (reg_alpha 1-8).** Single biggest hyperparameter effect:
   prunes noisy TE splits, 0.7392 -> 0.7434 (exp12), and per-member alpha diversity added
   +0.0017 (exp14: 0.7451).
5. **Two time clocks.** Cyclical minutes-since-midnight AND a "flight-day" clock
   (minutes since 5am, wrap at ~3am when the network shuts down) + ops_hour and
   origin/dest x ops_hour TEs (exp18/20: 0.7459 -> 0.7464). Dropping the midnight clock
   lost 0.0008 — both carry signal.

## 3 things that did not help

- **Day-level features (TE or count by exact calendar day, airport x day).** Catastrophic
  (-0.01): 2005 days never recur in 2006. Confirmed the year-shift framing — only
  physically-stable statistics generalize.
- **Deeper changes to learning rate / tree count.** lr 0.05 with rescaled trees (0.7304 vs 0.7367),
  10-fold OOF TE, dart booster, max_bin tuning, min_child_weight: all flat or worse.
- **Extra ensemble machinery.** 6th member (timeout at d16; 6-12 members flatlined at 0.7392 in
  probes), row-subsample bagging (0.7372), rank-averaging (= prob-avg), colsample/subsample grids.

## With more budget

CPU was the binding constraint, not ideas. Next: (1) tune the two clocks jointly — replace raw
`tod` integer with the flight-day clock in the slot TEs (slot30 keys use midnight boundaries);
(2) per-key alpha via hierarchical shrinkage toward the parent key's TE (route -> origin/dest)
instead of a fixed prior; (3) seed-averaged replicas of the best 5 members (3 seeds each) which
the flat 6-member result suggests only helps with much larger k; (4) light stacker: small d3
XGB on the 5 members' OOF-fold predictions, reusing the existing 5-fold structure for ~zero
extra cost; (5) calendar features that ARE year-stable: dow x month TE, day-of-month TE (both
were flat here, but only tested once with untuned alphas).
