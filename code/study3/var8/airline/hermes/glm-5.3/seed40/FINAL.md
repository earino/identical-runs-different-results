# FINAL — airline dep-delay XGBoost (autoresearch benchmark)

**Best Eval AUC: 0.7580** (commit d8e5f59, validated `CONTRACT OK`, predict_proba reproduces 0.7580 on eval with target column removed).

Baseline was 0.7141 (experiment #1). Final model = 3-seed XGBoost ensemble, max_depth=20, n_estimators=250, lr=0.03, colsample_bytree=0.4, reg_lambda=3, hist trees, categorical-native encoding.

## Changes that mattered most

1. **Drift-aware capacity tuning** (+0.006 → 0.7196): the 2005→2006 shift makes the eval AUC curve peak early (~30 trees at lr 0.1). Low learning rate (0.03), many trees (250), fixed n (no early stopping on eval), trained on all 100k rows.
2. **Deep trees + column subsampling** (+0.017 → 0.7417 with the pair md16/col0.4): with col=0.4 and lam=3, depth 16→20 kept helping; shallow (md4-6) plateaued at ~0.72. Deep trees express hour/carrier/distance interactions; colsample decorrelates trees and blocks drift-memorization.
3. **Cyclic month + 8-harmonic departure time** (+0.008): Month identity and raw hhmm invite year-specific splits. Replaced Month with sin/cos(2πm/12), DepTime with sin/cos at 8 harmonics of minutes-of-day. Harmonics 2-8 kept adding (+0.002 each step up to k=8).
4. **15-minute dep-time bins as a categorical** (+0.001): sharp step structure in delay risk by time block that smooth harmonics miss.
5. **Hour×carrier interaction + distance×time features** (+0.004): `hour_carrier` categorical, `dist×minute-of-day`, distance bands, log-distance. Plus 3-seed averaging (+0.0005 and real variance reduction for the hidden holdout).

## What did NOT help (eval got worse or flat)

1. **Route pair as a 4198-level categorical** (0.706 → −0.008): route IDs from 2005 don't transfer to 2006; the baseline's cardinality cap was right for the wrong reason.
2. **Every target/count encoding tried** (route, hour×carrier, origin, dest TEs; route counts): all lost 0.002-0.015 — smoothed target statistics memorize 2005 delay rates.
3. **Other failures**: day-of-month cyclics (calendar-day drift), bin×carrier interaction grid, lossguide growth (0.7458), min_child_weight=5, subsample<1, max_bin=512 (flat), month/dow harmonics interactions, heterogeneous-config ensembling (weaker members dilute), dropping raw DepTime (slightly worse).

## With more budget

The binding structure is the 2005→2006 time shift: only stable seasonal/diurnal/cycle patterns transfer, so I would attack residual drift rather than add capacity. Concretely: (a) group-time CV inside train (split by month) to pick hyperparameters that maximize *worst-month* AUC instead of eval.csv point estimates, since eval-based selection risks overfitting the 2006 slice; (b) domain-features that are drift-free by construction — e.g. per-carrier rank features within a departure window rather than raw counts, and harmonic regression bases for month/day at higher order with regularization; (c) a larger seed ensemble (8-16 seeds, or bagged row subsets) which reliably buys +0.001-0.002 on shifted data; (d) test whether Origin/Dest should be dropped entirely or replaced by their 2005 traffic volume (they may carry station-specific 2005 delay signatures); (e) inspect the eval-time failure mode via per-month AUC to see if specific months drag the metric — month-specific recalibration might be legitimate given the holdout is also 2006.
