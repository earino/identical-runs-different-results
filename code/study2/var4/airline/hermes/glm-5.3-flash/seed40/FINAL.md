# FINAL — airline XGBoost autoresearch

Best Eval AUC: **0.7465** (experiment #40, commit `d7639df`; validate.py: CONTRACT OK, 0.7465)
Baseline: 0.7141 → final: 0.7465 (+0.0324).

## Changes that mattered most

1. **Dropping `Month` entirely** (exp22, +0.0013 on top of the stack; verified by month-level delay
   rates shifting year-over-year: c-1 0.541→0.444, c-9 0.409→0.474). Month is the one feature whose
   train→eval mapping is unstable across the 2005→2006 split; removing it let every other feature
   contribute more. A custom time-ordered CV (train past → validate future) was built to detect
   exactly this kind of shift-sensitive feature.
2. **Quarter-of-day categorical** (`dep_tod_cat`, 96 levels) + **day-of-month categorical**
   (exp31, +0.0100 — the single largest jump). Scheduled time affects delay non-monotonically
   (ground-ops congestion waves), and trees with `enable_categorical` exploit per-level effects.
   Coarser hour-cat (24 levels) alone was worth +0.004 earlier (exp3).
3. **Regularized deep trees** (exp10, +0.0058): max_depth 11, eta 0.02, min_child_weight 20,
   lambda 2, alpha 0.5, 800 rounds. Shallow/mid configs (exp9) and depth 9 (exp28) lost.
4. **2-model ensemble** (exp29, +0.0017): base member + depth-14/subsample-0.9/mcw-50 member
   (exp40, +0.0001 more). Diverse-depth averaging beat a 3-seed ensemble (exp15, tie) and
   bootstrap-bagged variants (exp30/33/36, all equal-or-worse at 3x cost).
5. **Smoothed target encoding** of Origin/Dest/UniqueCarrier, fit on train only with alpha=20
   (exp4, +0.0005). alpha=5 was worse (exp24); raw categorical airports alone underuse per-airport risk.

## Things that did not help

- **Distance engineering** (log, 7-bin categorical, exp5) and **hour×distance / hour×carrier
  numeric interactions** (exp8): all neutral-to-negative.
- **Holiday/season flags** (near-Thanksgiving/Christmas, weekend, exp6) and **time-of-day bucket
  dummies** (exp18): redundant with the categorical calendar features.
- **Frequency encodings** (exp20), **cat_smooth=50** (exp17), **lossguide/max_leaves=255** (exp27,
  timed out), **5-min buckets + bad-DepTime flag** (exp34), **dow×tod 672-level interaction** (exp38):
  no gains over the simpler stack.
- **Route TE / origin-hour TE** (exp13): hurt (−0.009) — sparse maps overfit 2005 patterns.
- **o-d pair-ID feature** (exp14): −0.0004.
- Random **CV-in-train harness** printed good numbers but consumed 2/3 of runtime per experiment;
  removed from the timed path at equal AUC (exp37, 34s vs 64s).

## With more budget

- Grade the dow×tod and origin×carrier interactions with more rows (2005 has ~7M flights; the 100k
  slice is thin for 672-level effects).
- Tune the second member independently (its mcw/lambda were copied once and never revisited).
- Early stopping per member against a time-ordered split instead of fixed 800 rounds.
- Quantile-bucketed DepTime at 2/3-minute resolution as categorical, tested one at a time —
  the 96-level tod cat was the biggest win, finer grids may add more.
- Isotonic/rank calibration of the averaged probability (AUC is rank-based; averaging logits vs
  probs was never tested).

## What would break the contract (avoided)

- Any train/eval-level transformation outside `prepare()` — all engineering lives there, levels and
  TE maps are fit on `train` only, unseen categories map to missing (validated by `validate.py`).
