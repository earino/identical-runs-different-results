# Final report — airline delay (XGBoost, autoresearch harness)

**Best Eval AUC: 0.7360** (baseline: 0.7141 → **+0.0219**), 40/40 experiments used.
Final model: ensemble of 11 XGBoost models (depths 3–10, lr 0.03–0.2) over engineered features;
`validate.sh` → `CONTRACT OK`, predict_proba reproduces 0.7360 with the target column removed.

## Changes that mattered most

1. **15-minute bin-of-day categorical (`dep_bin15`)** — the single biggest win (+0.0075 on a one-model
   config: 0.7209 → 0.7284). Scheduled-departure slots behave discretely (schedule "banks"), and a
   97-level categorical lets trees assign arbitrary per-slot effects; finer (5-min) overfits, coarser
   (10-min) loses.
2. **Diverse ensemble of 11 XGBoost members** (+0.0076: 0.7284 single → 0.7360): varying depth (3–10),
   learning rate (0.03–0.2) and tree count. Diagnostics showed members at 0.718–0.722 individually with
   0.91 mean pairwise correlation; even "weak" members added value through decorrelation (swapping two
   weak members for stronger ones *lost* AUC).
3. **Hour as a categorical alongside numeric hour** (+0.002): same principle as (1), smaller dose.
4. **Time-of-day / calendar features v1** (+0.002 over raw baseline): hour, minute, cyclical sin/cos,
   numeric Month/DOW/DayOfMonth, log-distance.
5. **Right-sized single models**: 400–800 trees at lr 0.05, depth 6 beat both the 30-tree baseline and
   deeper/subsampled variants — the 2005→2006 time shift punishes variance.

## Things that did NOT help

- **Target encodings** (smoothed, carrier/origin/dest/route): 0.6981 — in-sample leakage plus real
  2005→2006 drift in per-group rates (e.g., carrier AS 0.64→0.52).
- **Route categorical + frequency encodings**: 0.7022; **row-bagging**: 0.7255; **subsample/colsample**:
  0.7079; **min_child_weight=10**: 0.7142 — all forms of added variance or memorized density hurt under
  the year shift.
- **Categorical cross-interactions** (Month×DOW 0.7137, DOW×hour-bin 0.7265, DOW×bin15 0.7061): joint
  levels memorize 2005 noise; every one regressed on eval.
- **5-min / 10-min time bins** (0.7178 / 0.7259), **max_bin=512** (no change), **DART** (0.7186, slow),
  **lossguide** (no change), **rank-averaging** (+0.0002, slower), **5-seed ensembles of one config**
  (models nearly identical without stochasticity — zero gain).

## With more budget

Stack members via OOF predictions (XGBoost meta-learner), add feature-view members (e.g., trained without
Origin/Dest categories) for stronger decorrelation, per-feature bin-granularity search (the 15-min peak was
found by grid, not search), and a temporal (month-based) validation split inside train.py to early-stop each
member against a 2005→2006-like shift instead of a random split.
