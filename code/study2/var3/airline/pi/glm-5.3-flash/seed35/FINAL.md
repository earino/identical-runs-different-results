# Final Report — airline delay XGBoost (autoresearch)

**Best Eval AUC: 0.7373** (baseline 0.7141, +0.0232). Final commit `55d1bbc`, validate.sh: CONTRACT OK.

Final model: 20-model XGBoost ensemble in two configs — 10× (300 trees, lr 0.05, depth 6, subsample 0.6,
colsample 0.6) + 10× (150 trees, lr 0.1, depth 6, subsample 0.85, colsample 0.85), all with gamma=1 and
native categorical support. Features (all inside `prepare()`, encoders fit on train only): numeric
month/day-of-month/day-of-week, 15-minute scheduled-departure bucket (categorical), hour (numeric +
categorical), log-distance, holiday-window flag (fixed-date + floating US holidays, derived from
month/dom/dow only), and traffic counts (origin, dest, route, carrier).

## Changes that mattered most

1. **Bagged ensemble instead of a single model** (exp14→19: 0.7198 → 0.7268). Single-model tree count
   saturates/overfits almost immediately under the 2005→2006 shift (30 trees = 0.7141, 1000 trees =
   0.7004), but 12–20 strongly subsampled members keep improving. Largest single lever (+0.013).
2. **15-minute scheduled-departure bucket as categorical** (exp27–28: 0.7275 → 0.7347, +0.0072).
   Schedule granularity (hhmm // 15) is the strongest feature in the dataset; finer (10-min) or coarser
   (30-min) buckets were both worse. Raw DepTime as numeric severely underuses this signal.
3. **Time/calendar feature set** (exp10–11: 0.7141 → 0.7166): numeric Month/DayofMonth/DayOfWeek,
   hour numeric + categorical, log-distance. Small but robust, and the base for everything after.
4. **Traffic counts + holiday flag** (exp30/33/23: 0.7347 → 0.7357): origin/dest/route/carrier row counts
   (hub size, structurally stable across years) and a year-stable holiday-window flag built purely from
   calendar arithmetic.
5. **Two-config ensemble + gamma=1** (exp38/40: 0.7367 → 0.7373): mixing differently-biased members
   (slow/shrunk vs fast/loose) and split-loss regularization added the last ~0.0006.

## What did not help

- **Target encoding** (smoothed m=50 on carrier/origin/dest/route/hour/dow + counts): 0.7059 vs 0.7166 —
  2005 group delay rates do not transfer to 2006; explicit memorized statistics hurt under the time shift.
- **Raw Origin_Dest route categorical** (~6k levels): 0.7076 — trees memorize route-specific 2005 noise.
- **Bigger/deeper single models, early stopping on an internal random split, per-member bootstrap,
  depth-diverse members, colsample_bynode, month/dow as extra categoricals, min_child_weight** — all
  neutral or negative (the internal-split early stopping stopped at 182 trees and lost to 30 trees).

## With more budget

I would (a) build a proper time-aware validation protocol (train on 2005-H1, validate on 2005-H2) to make
keep/discard decisions measure shift-robustness instead of in-year AUC, since single-run eval deltas below
~0.002 are likely noise; (b) explore schedule-structure features further (per-carrier×half-hour schedule
position, connection/turnaround proxies from route frequency); (c) run a small randomized search over the
ensemble config grid (lr × trees × subsample) with 3-seed averaged evaluation; (d) try snapshot-style
members trained on progressively larger time slices of 2005.
