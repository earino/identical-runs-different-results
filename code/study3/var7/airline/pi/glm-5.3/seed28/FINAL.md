# Final report — airline dep-delay-15min, XGBoost autoresearch

## Best result

- **Best Eval AUC: 0.7649** (experiment #9, commit `137b926`, reproduced by `validate.sh`
  through `predict_proba` with the target column removed — `CONTRACT OK`).
- Baseline (committed starting point, 30 shallow trees, raw categoricals): **0.7141**.
- Net improvement: **+0.0508 AUC**.

## Changes that mattered most (in order of impact)

1. **Deep trees + heavy feature bagging + slow learning.** Replacing the 30-tree/depth-6/lr-0.1
   regime with `max_depth=24`, `colsample_bytree=0.3`, `learning_rate=0.02`, 200 rounds moved
   eval AUC from ~0.718 to ~0.754. On this time-shifted split (train 2005 → eval 2006) capacity
   in the *conventional* sense (more trees, full features) badly overfits; capacity in the
   *random-forest* sense (deep trees, strong per-tree feature subsampling, slow averaging)
   generalizes. This one regime change dwarfed everything else.
2. **Numeric time/calendar features.** `c-N` strings parsed to ints; `DepTime` → hour, minute,
   minutes-since-midnight; numeric month/dom/dow; log-distance. Replacing the string categories
   with numeric scales was worth about +0.004 on its own.
3. **Redundant encodings of the strong time features** (`minute`, `tmin/60`, `hour2`, `dep2`
   alongside hour/tmin/DepTime). Under colsample 0.3, duplicating the few strong features raises
   the probability that any given tree can actually split on time-of-day (+0.003). Dropping
   either of the near-duplicate pairs (DepTime/tmin) cost 0.011 AUC.
4. **Label-free schedule-density counts**, fit on train only: log1p counts for
   origin/dest/route/carrier × hour (+~0.002). These encode "how busy is this airport at this
   hour", which is a stable, structural property of the schedule that survives the year shift.
5. **3-seed ensemble with bin-granularity diversity** (members with max_bin 256/384/512,
   otherwise identical): +0.005 over a single model (0.7627 → 0.7649). Plain seed diversity
   gave +0.0025; depth/colsample/row-bootstrap diversity gave nothing.

## Things that did NOT help (negative results, each measured)

1. **Target/impact encoding** of route/carrier/origin/dest (any smoothing m=20..100, with or
   without count features): consistently **-0.01**. Per-key delay *rates* fitted on 2005 do not
   transfer to 2006 — the single most instructive failure of the session. Only structure
   (schedules, density, systematic hour-of-day effects) generalizes across the year shift.
2. **High-cardinality native categorical `Flight` (Origin+Dest)**: -0.02. Also cyclical
   sin/cos encodings, day-of-year seasonality, holiday-distance features, ratio features,
   month/dow count variants, interaction count features — all flat or negative.
3. **More/other capacity**: >200 rounds, lr>0.02, min_child_weight>1, reg_lambda/alpha/gamma,
   row subsampling, colsample_bynode/bylevel, lossguide growth, DART (also 4x slower),
   raw (non-log) counts — all flat or worse. Row-bootstrap ensembles lost 0.0025.

## Robustness notes

- A **temporal sanity check inside train** (fit months 1–9 2005, validate months 10–12 2005,
  all statistics fit on the early slice only) gave 0.7893 for the final config vs 0.7030 for
  the baseline config — the gains are not eval.csv-specific tuning.
- Run-to-run AUC noise of the final config is ~±0.0015 (multithreaded hist); the last 8
  experiments all landed on the same plateau (0.7621–0.7650). Final config runs in ~86–89s of
  the 120s limit; a 4-member variant that measured 0.7650 (+0.0001, pure noise) was rejected
  because it needed 101s and the sub-0.0005 gain was not plausibly robust.

## What I would try with more budget

The binding constraint was the 120s wall-clock cap per run, not ideas. First I would build a
16–32-member ensemble (varied seeds, max_bin, colsample in 0.25–0.35, maybe per-member column
order) — ensemble variance reduction clearly worked, and member count beyond 3 was purely
compute-capped; with the cap lifted this likely adds a further +0.002–0.005. Second, I would
revisit per-key statistics that are *year-stable by construction*: instead of 2005 delay rates
(which shift), schedule-table features aggregated from a larger unlabeled flight universe
(route frequency by hour of day, fleet/turnaround proxies), or TE with explicit year-shift
shrinkage toward the carrier/airport hierarchy. Third, stacking: OOF predictions from several
diverse XGB families fit on 2005 folds with a tiny logistic/XGB meta-learner, selected on a
late-2005 temporal split to avoid eval overfitting. Finally, I would spend CPU on a proper
hyperparameter search *around the found ridge* (depth 20–28 × colsample 0.25–0.35 × max_bin
256–512 jointly), since the regime was found by coordinate ascent and interactions may hide
another few thousandths.
