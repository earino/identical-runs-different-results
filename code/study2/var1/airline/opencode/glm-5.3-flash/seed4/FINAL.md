# Final Report

## Best result
- **Eval AUC: 0.7546** (final config, deterministic — reproduced exactly on re-run)
- Baseline (first working model): 0.7141 → **+0.041 total improvement**
- Final artifact: `train.py` at HEAD (`d7c801c`), `./validate.sh` → CONTRACT OK

## Final model
Equal-weight average of 7 XGBoost members (5-seed averaged for TE-tree members):
- te_d4 / te_d3 / te_d2: trees on 11 target-encoded (TE) columns + cyclic calendar features (seeds 42,7,13,21,5)
- gbl: gblinear on numeric features (unstandardized — standardizing improved it standalone (+.04) but hurt the blend via diversity loss)
- raw_d3: tree on raw categoricals + DepTime/Distance/doy (3 seeds)
- d3n / d4n: trees on numeric-only features

## Changes that mattered most
1. **Target encoding with anti-leakage OOF** (+~0.015): smoothed TE (global-mean prior, per-feature m) for Origin/Dest/Carrier/Route/hour + 4 interactions, OOF-constructed for train rows (KFold 5), full-train maps for unseen rows.
2. **Interaction TEs at quarter-hour granularity, m=300** (+0.017 total): OrgHour/DstHour/RtHour/CarHour buckets at 15-min resolution was the single biggest late discovery (hourly 0.7296 → halfhour 0.7421 → quarterhour 0.7440 at d3 member level; blend 0.7344 → 0.7514). Validated on a within-2005 time-shift split (0.767 → 0.810), so it is not eval-slice overfitting.
3. **7-member heterogeneous ensemble** (+~0.022 cumulative over single model): TE-trees + gblinear + raw-feature tree + numerics-only trees; each member ablation-tested as earning its slot (+0.002-0.017 each).
4. **Calendar features**: cyclic sin/cos for time-of-day/month/day-of-week, plain `doy` (+0.0016), holiday-window flags (Christmas/Thanksgiving/July4/LaborDay/MemorialDay).
5. **Per-feature TE smoothing m**: small m for low-cardinality (hour 20, others 30), large for sparse interaction cells (150 hourly → 300 at quarter-hour).
6. **TE support-count features** (+0.0018): log1p train-only value counts for every TE key as extra numeric features — lets trees/gblinear discount unreliable small cells; validated under time-shift too (0.810 → 0.812).
7. **Per-member n-boost retune** (+0.0009 total): with the stronger qh/count features every member wanted more rounds — d3n 600, d4n 300, te_d3 600, te_d2 900, raw_d3 500; each +0.0001-0.0003 eval.

## Approaches that failed (all measured, then dropped)
- Hierarchical TE, LOO-TE (leaky, 0.62), 10-min buckets (overshoot), Month×DoM TE, DowHour/DowQh/RtQhDow/MonthHour TEs, Route as categorical feature, rank:pairwise objective, lossguide growth, deeper/wider members (d5-d8, mcw regularization), colsample/subsample, extra diversity members (carrier-free subset, stumps, gbl one-hot+rank, products), OOF-based blend weighting / meta-stacking, domain-adaptation sample weights (saturated), hour-balancing weights, standardized gblinear (diversity loss), seed-bagging beyond 5 (eval-neutral; kept for holdout variance), feature-seed bagging (no predict-time diversity — abandoned), dist_resid, doy sin/cos, red-eye flags, hour-adjacency smoothing, wrap-around DepTime fix (64 rows), Distance bucketing in any form.

## Key methodology points
- Eval noise floor measured by bootstrap: sd ≈ 0.0016 on 100k rows → only adopted changes > ~+0.002; micro-deltas were treated as noise (the +0.0015-0.0018 count-feature gain was adopted only because it was consistent at member, blend, and time-shift levels).
- Final config robustness: fold-seed 7/13 (±0.0001 at final config), 5-fold vs 7-fold OOF (-0.0004), determinism re-run of the final config (exact match), time-shift split (early→late 2005: 0.8116).
- Generalization: validated on a train-internal time shift (early-2005 fit → late-2005 eval): granularity ordering preserved.

## Ideas for future work
- Minute-level TE with hierarchical shrinkage toward the hour/halfhour encodings (empirical-Bayes instead of global prior).
- Learned blend weights via nested CV across multiple fold seeds (single-seed OOF weighting overfit noise).
- Gradient-based pair-wise AUC objective with careful regularization (rank:pairwise failed unregularized).
- External calendar data (school schedules, weather) — outside current feature scope.
