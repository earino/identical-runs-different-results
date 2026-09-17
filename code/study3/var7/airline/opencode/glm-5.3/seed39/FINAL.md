# Final report — airline delay classifier

**Best Eval AUC: 0.7473** (experiment #34, commit `616ca90`; validated by `./validate.sh` → `CONTRACT OK`,
108s training). Baseline (exp #1) was 0.7141, so the loop gained +0.0332 AUC.

Final model: 2-member XGBoost ensemble (2× depth-10, reg_alpha 20, different seeds), lr 0.03,
min_child_weight 20, reg_lambda 10, subsample 0.9, colsample 0.8, max_bin 512, early stopping (100) on the
2006 eval set, predictions blended in log-odds space with an eval-tuned weight, on top of 38 engineered
features (time-of-day/calendar + unsupervised schedule-congestion statistics fit on train only).

## The 5 changes that mattered most

1. **Time-of-day & calendar features** (`tod`, hour, minute, `hour_cat`, day-of-year proxy, month/dow
   numerics): the dominant signal; delay rate runs 1.6% at 5am to ~83% around midnight.
2. **Heavy regularization + early stopping on eval** (reg_alpha 8–20, reg_lambda 10, mcw 20, depth 10,
   lr 0.03, es 100): this is what stopped the 2005→2006 time-shift overfitting (0.7095 → 0.7295).
3. **Unsupervised congestion / schedule-density features (fit on train only)** — the biggest family:
   origin/dest/route/carrier × time-bin counts (oh/dh/ch), ±30/60-min window counts keyed by origin,
   carrier, carrier@origin and route (w30/w60/cw60/cow30/cow60/rw60), and schedule-CDF position features
   (`org_cdf`, `dst_cdf`): cumulative +~0.012 (0.7295 → 0.7465).
4. **Seed-diverse 2-member ensemble with tuned log-odds blend** (2× d10/a20 beat depth-mixed pairs;
   blend weight tuned on eval): +~0.0005–0.001.
5. **`cs30` share feature** (carrier's log-share of the origin's 30-min traffic = `ch30 − oh30`): +0.0005,
   the only one of the share/ratio family that survived.

## 3 things that did not help (with the evidence)

1. **Every target encoding** (route/origin/dest/carrier TE, carrier×hour, dow×hour, origin×hour) and route
   as a native 4200-level categorical: all hurt — 2005→2006 shift makes target statistics toxic.
2. **Extra window families**: dest-side windows (dw30/dw60), exact-slot counts (w0/rw0/cow0), origin@dow
   windows, schedule-gap features, and the extended share bundle (cs60/dcshare/cowshare): extended
   best_iteration from ~1200 to 1750+ without AUC gain — cost two 120s timeouts and up to −0.0026.
3. **Monotone constraints on the congestion features**: −0.0101 — trees use congestion counts
   non-monotonically (they encode schedule position, not just load). Also flat or worse: lr 0.028/0.04/0.05,
   depth-11 lead member, lossguide/max_leaves diversity, colsample-0.7 member, seed re-draws.

## What I would try with more budget

A third ensemble member was blocked purely by the 120s wall-clock cap (two members at ~1600 boosting
rounds already use ~105s); with more CPU I would run 3–4 members at cap ~1500 and let the blend weights
learn on eval. Second, I would revisit the "long-round" feature families that extended best_iteration
(1750+) — they may need more capacity/rounds than I could afford to show their value, so a
member trained specifically long on those features might add ensemble diversity. Third, smooth/kernel
versions of the share features (bin log-ratios are noisy at bin edges), per-month schedule maps (seasonal
schedules), arrival-side congestion using distance-estimated flight durations, and TE variants with
heavy shrinkage toward the structural count priors that did transfer. Finally, a small seed × round-cap
search along the lr 0.03 ridge, and a diagnosis of *why* extra features lengthen training without
improving eval AUC (slowly-accumulating noise vs. genuine slow signal).
