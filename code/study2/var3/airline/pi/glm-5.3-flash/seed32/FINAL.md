# Final Report — airline delay AUC (XGBoost, autoresearch harness)

**Best Eval AUC: 0.7248** (baseline: 0.7141, +0.0107) — 40/40 experiments used.

Final model: a 6-member XGBoost ensemble (`xgb.train`, hist, max_bin 256, lr 0.04, ES on a
random 40% validation split of train, refit on full train at `best_iteration+1`), averaged
over members with diverse depth/regularization and two target-encoding variants (M=50 and
M=150 smoothing). All feature engineering lives in `prepare(df)`; TE maps are fit on
train.csv only.

## Changes that mattered most

1. **Target-encoded hour interactions** (exp7–9, +0.004 over baseline single model):
   TE of DepHour, DepHour×DayOfWeek, DepHour×Month, Origin×Hour, Carrier×Hour — plus
   DepHour as a categorical. Delay risk is governed by scheduled time-of-day and its
   seasonal/weekly modulation; these dense features let small trees exploit it.
   Selection rule: keep only TEs whose AUC is stable from train (2005) to eval (2006).
2. **Ensembling** (exp14, 0.7203 → 0.7227): 5 diverse members (depth 6/8/10, mcw, colsample,
   subsample, different ES splits) beat the best single model by +0.0024.
3. **TE-smoothing diversity across members** (exp21, → 0.7231): members trained on two TE
   variants (M=50 vs M=150) decorrelate errors — a robustness play for the year shift.
4. **Validation-protocol tuning** (exp23/28, → 0.7244): ES validation fraction 0.2 → 0.4 gave
   better early-stopping counts; ES50 patience beat ES30.
5. **Member curation** (exp25/38, → 0.7248): replacing weak members (subsample 0.8, depth 10)
   with clones of the strongest config (d8/mcw10–1, col 0.8–0.9) lifted the ensemble mean.

## What did not help (reverted)

- **Generic FE**: cyclical sin/cos of month/DOW/DOM, log-distance, DepTime minute features —
  dilute a small tree budget (exp3/4/5); Route categorical and Route TE transfer badly across
  the 2005→2006 year boundary (exp3/6: 0.70 under baseline).
- **Residual (two-layer) target encoding** for Origin×Hour / Carrier×Hour — better standalone
  AUC (0.699 vs 0.688) but worse in-model (0.7227 vs 0.7244); the model already had the base.
- **gamma=2.0** (0.7201 — collapses tree counts), **max_bin 128/192** (−0.0005), **ES30**
  (−0.0006), **8 members @ lr 0.05** (0.7222), **weighted-by-val-AUC averaging** (no gain),
  **more ES trees via EvaluationMonitor period trick** (broke early stopping, 0.7214),
  **warm-start continuation instead of refit** (slower, timed out).

## With more budget

- Scale the ensemble to 12–16 members (time-capped at 6 members ≈ 100 s here) with a wider
  smoothing spectrum (M ∈ {25, 50, 100, 200, 400}) and depth {6, 7, 8, 9}.
- True K-fold OOF stacking of member predictions with a shallow XGBoost meta-learner
  (infeasible in 120 s per run at this member count).
- Per-member feature views (e.g., one member with DepTime as a 1341-level categorical,
  one without the weak Origin/Dest TE pair) for cheaper diversity.
