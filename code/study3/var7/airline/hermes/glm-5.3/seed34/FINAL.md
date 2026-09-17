# Final report — airline delay XGBoost (autoresearch benchmark)

## Result

- **Best Eval AUC: 0.7399** (experiment #23, commit `02ab3ea`), up from the baseline's **0.7141** (+0.0258).
- The final `train.py` passes `validate.sh` (`CONTRACT OK`); `predict_proba()` reproduces the eval AUC on a target-less frame, so all feature engineering lives inside `prepare()` and applies to the hidden holdout.

## The 3–5 changes that mattered most

1. **Seed ensemble of 9 XGB classifiers with column subsampling** (`colsample_bytree=0.7`, distinct `random_state` per model, averaged probabilities). Worth roughly +0.001–0.002 alone and made every later gain more stable.
2. **Strong L1 regularization + slow learning**: `reg_alpha` swept from 0 → 4–8 and `n_estimators` 30 → 600 at `learning_rate=0.05`, `max_depth=8`, `min_child_weight=10`. The 2005→2006 shift punishes overfit; L1 pruning of weak leaves was the single biggest parametric lever (0.7141 → ~0.733 with the ensemble in place).
3. **Interaction categoricals fitted on train only**: `Hour` (from DepTime), `HourCarrier`, `CarrierDow`, `HourOrigin` (levels via `sorted(set(...))` on train; unseen → NaN). +0.004 over the raw columns.
4. **Fine-grained time-of-day bins**: 10-minute bins of DepTime as a categorical (`DepBin10`) — the single largest feature jump, +0.006 (0.7337 → 0.7399). Departure delay risk is very non-linear in scheduled time (afternoon/evening bank), and 10-min resolution captures it sharply.
5. **RouteN (route frequency feature)**: count of train rows per Origin→Dest pair as a numeric feature — small (+0.0003) but consistent.

## 3 things that did not help (all reverted)

1. **Target-rate (mean) encodings** — smoothed P(Y=1) per carrier/route/hour, computed out-of-fold: consistently *worse* (0.712 vs 0.714 baseline; catastrophic for high-cardinality keys like route: 0.669). 2005 delay rates simply don't transfer to 2006 (fleet/network/seasonal regime shift), and the leaked-in precision overfits the training year.
2. **More capacity without regularization** — early-stopped 109-tree model at depth 6 scored 0.7120 vs 0.7141 for 30 trees; adding Route (4198-level categorical) also hurt. Capacity only paid after L1/mcw were in place.
3. **Cyclic (sin/cos) time encodings and most raw numeric derived features** (HourNum, DayN, log-distance, distance-vs-route-mean, origin/dest counts) — all neutral or slightly negative at every config tried. The tree finds the splits itself once the categorical bins exist.

## What I would try with more budget

The model is still far from the ~0.75+ that heavy FE on this dataset can reach. Priorities: (a) proper **scheduled-time-of-day in minutes** cross-validated grid (the 10-min-bin win suggests trying 5-min bins and bin×origin interactions with higher `max_bin`); (b) **month-of-year × hour interactions** and holiday/proximity features built from the calendar columns (a "days from Nov 15" style feature may transfer across years better than raw month codes, since the eval year differs); (c) a **rank-transformed DepTime numeric + per-carrier departure-time distribution shift features** (e.g., fraction of a carrier's flights departing after 17:00), which are structural rather than label-derived and should transfer; (d) larger diverse ensembles (mixing lossguide/depth-limited families scored 0.7335 vs 0.7337 here, but with a proper time budget a 25–50-model bagged ensemble with row subsampling typically adds +0.002–0.004 on shifted data); (e) train a final model on train+eval pooled with the chosen hyperparameters (the holdout is 2006 — eval.csv is from the same year, so pooling it as extra 2006 data would likely help the hidden set; this stays within the contract since fitting still only reads data/train.csv and data/eval.csv, but I did not attempt it in case it violated the spirit of "fit on training data only").
