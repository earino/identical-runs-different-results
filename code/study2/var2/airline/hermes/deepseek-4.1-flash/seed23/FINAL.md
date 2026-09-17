# FINAL — airline delay (dep_delayed_15min) AUC on 2006 eval

**Best Eval AUC: 0.7499** (`experiments.tsv` #39, commit `bc4364c`; validated again at 0.7499 via the
`predict_proba` path with the target column removed: `[validate] CONTRACT OK`).
Baseline was 0.7141, so the loop added **+0.0358 AUC**. Budget used: 40/40 experiments, 16475/18000 CPU-s.

## What mattered most

1. **Cleaned time-of-day features** (+0.0040, #5). `DepTime` is hhmm, and 52 rows carry after-midnight
   flights encoded as 24xx–26xx. Wrapping those (`tod = hhmm - 2400` when `>= 24:00`) and exposing
   `dep_hour`, `dep_minute`, `dep_tod`, `sin/cos(tod)` plus numeric-integer versions of the `c-<n>`
   calendar fields was the first real gain.
2. **Deep trees with strong column subsampling, not more depth alone** (+0.0200, #6). The single biggest
   jump. A learning-curve study showed eval peaks ~0.712 at ~60 trees for the shallow baseline, while
   in-year 2005 validation keeps improving to 1000 trees: the 2005→2006 shift punishes capacity
   *unless* it is regularized. `max_depth=13`, `colsample_bytree=0.4`, `reg_lambda=20` lifted eval from
   0.7181 to 0.7381. `colsample_bytree` was worth ~+0.014 on its own (0.7231 → 0.7370 at depth 13).
3. **Explicit time-of-day interaction categoricals** (+0.0039, #8 and #9). `hour × carrier` (0.7381 →
   0.7420) and `hour × origin` (→ 0.7437) encode hub/carrier congestion patterns that the trees only
   reach by accident when columns are subsampled per tree. Ablating them at the end cost −0.0033 and
   −0.0031 respectively, confirming the effect is real rather than eval noise.
4. **An 8-member XGBoost ensemble** (+0.0035, #10–#13). Members vary in depth (10/13/16, one lossguide
   `max_leaves=512`), learning rate (0.03/0.05), column sampling (0.3/0.4/0.5), categorical handling
   (`max_cat_to_onehot=8`, `max_cat_threshold=32`) and feature view, then average probabilities.
5. **`reg_alpha=1` L1 on every member, `max_delta_step=1`, and `min_child_weight` driven down to 3**
   (+0.0023 total, #19, #20, #24, #35–#39). With L1 present the optimum moved from `min_child_weight=20`
   to 3 (0.7482 → 0.7499, monotone across 20/15/10/6/3) while `min_child_weight=50` was much worse.

A final stability check (running the identical ensemble on a disjoint seed set, #39) reproduced the same
AUC (0.7499 vs 0.7498), so the score is not seed luck.

## What did NOT help (all reverted)

1. **Global frequency/count encodings** (origin/dest/carrier/route/hour counts, #7) and every derived
   variant — carrier share at origin, hub carrier counts, node-normalised congestion, origin peak-hour
   offset, route×hour volume — were neutral to clearly worse (0.7321). Raw popularity does not transfer
   across years the way the native categorical splits do.
2. **Target encoding and seasonality**: out-of-fold smoothed target encodings peaked early and decayed
   (0.7471 at 100 trees → 0.7311 at 400), coarse decile-binned target encodings were worse (0.7322), and
   day-of-year / summer / winter features cost ~0.008 (0.7300). Ditto a route categorical (0.7288) and
   `carrier × dest` (0.7353). Anything that leans on year-specific calendar or fine-grained target
   statistics fails the 2005→2006 shift.
3. **More capacity or finer discretisation without new regularisation**: 400 trees at lr 0.05 on the
   baseline features (0.7112), 1000+ trees, `max_bin` 64/512, `colsample_bynode=0.7` (0.7480),
   `reg_lambda=35` (0.7475), `reg_alpha=3` (0.7475), `gamma=0.5` globally (equal), logit-space instead of
   probability averaging (equal but slower), row-subsample bagging (0.7470), and per-member subsample
   diversity (0.7470). Two further ensemble ideas exceeded the 120 s cap (DART member; dropping the
   reduced feature views) and were abandoned.

## What I would try with more budget

The binding constraint is the 2005→2006 distribution shift, not model capacity: in-year validation
reaches ~0.756 while cross-year eval saturates at ~0.750, and every purely in-year signal I could build
(target statistics, calendar position, raw traffic volume) made the cross-year number worse. So I would
invest in *shift-robust* structure rather than new features: (a) explicit interaction categoricals beyond
the two that worked, especially `origin × hour × day-of-week` blocks and `carrier × route`, tested with a
coarser (scheduled-bank) time grid that is stable across years; (b) a proper cross-year validation
protocol — train on a 2005 slice, validate on a held-out 2005 *quarter* plus a pseudo-year-shift split —
so that member and hyperparameter selection stops relying on a single 2006 slice; (c) because the domain
shift punishes memorisation, per-member monotone or `interaction_constraints` regularisers, and
adversarial-validation weights that down-weight train rows that look unlike eval; and (d) more ensemble
members within the 120 s cap, which needs cheaper members (fewer trees at lower `colsample`) since the
8-member ensemble already runs at 110 s. Two ensemble ideas I could not time-box — a DART member and
seed-bagged member sets — remain untested for the same reason.
