# FINAL — airline delay prediction (XGBoost), 40/40 experiments

## Result

| | Eval AUC |
|---|---|
| baseline `train.py` (30 trees, depth 6, raw features) | 0.7141 |
| **final `train.py`** (commit `b6a3ba4`) | **0.7660** |
| validation (`./validate.sh`, drop target, `predict_proba`) | 0.7660, `CONTRACT OK` |

Budget closed at 40/40 experiments (29 kept, 11 reverted), ~17.0k of the 18.0k CPU-seconds, no crashes,
timeouts or OOM kills. `HEAD` is the best-Eval-AUC commit; `./validate.sh` re-runs training and confirms
`predict_proba` reproduces the same AUC on `data/eval.csv` with the target column removed.

## Setup that produced the result

`prepare()` is the only feature path (used for fitting *and* for scoring unseen rows). It builds
engineered time features (hour, minute, tod, tod-sin/cos, hour-as-category), numeric calendar fields,
`Distance`/`log_dist`, native categorical `UniqueCarrier`/`Origin`/`Dest`, and eight **smoothed
target-encoding columns** ("rate tables") keyed on `{origin, dest, carrier, carrier×origin,
carrier×dest, carrier×origin×dest} × 15-minute departure bucket` (the last two also at a 30-minute
bucket). Rate tables are fitted on training data only; the training matrix uses **out-of-fold** values
(2 folds) while scoring uses the full-training map. The model is a bag of 16 `XGBClassifier` members
(12 with the rate tables: 300 trees, depth 5, `colsample_bytree=0.5`, `reg_alpha=5`; plus 4 members that
never see the rate tables, depth 4/5/6/8, `colsample_bytree=0.8`), averaged with weight 1.5 on the
rate-table-free members.

## The 5 changes that mattered most

1. **Time-interacted rate tables** (`+0.0519` cumulative, by far the largest lever). Iterating the key
   from `origin×hour` → 30-min → 15-min bucket `+0.0314`, then adding `carrier×origin×15min` `+0.0129`,
   `carrier×dest` `+0.0092`, and `carrier×origin×dest×15min` (= the recurring daily flight slot, the
   strongest single feature) `+0.0084`. A cardinality-matched **random-key control** and a
   **shuffled-label control** both came back far worse (0.7470 / 0.7109 vs 0.7607 on the temporal
   split), so this is real signal, not leakage from the encoding machinery.
2. **Engineered time/calendar features + shallow trees** (`0.7141 → 0.7228`): raw `DepTime` and
   `c-<n>` strings replaced by hour/tod/cyclic features and numeric calendar fields; depth 6 → 3-5.
   The intraday ramp is the dominant single signal (delay rate 0.04 at 05:00 vs 0.82 at 23:00 and the
   same shape in both years).
3. **Out-of-fold count = 2 instead of 5** (`0.7541 → 0.7585`, and the same direction on the temporal
   split): noisier training-side rates stop the model over-trusting the rate tables, while scoring gets
   the sharp full-data map. Adding coarse (30-min) duplicates of the two strongest keys gave another
   `+0.0031`.
4. **Ensemble diversity, not just seeds** (`0.7589 → 0.7621`): adding four members that never see the
   rate tables (they read the raw schedule directly) is where the bagging gain came from; extra
   same-family seeds added ~nothing, and up-weighting the decorrelated members added a little more.
5. **L1 regularisation + retuned depth** (`0.7621 → 0.7660`): `reg_alpha` 1 → 5 improved `+0.0012`
   then `+0.0023` (15 was worse), and with sparsity shrinking the leaf weights, depth 4 → 5 became
   better instead of worse. This is the pairing that the unregularised model could not exploit.

## The 3 things that did not help (all reverted)

1. **More capacity without sparsity**: deeper trees (6/8), more trees (up to 1200), 1000+ trees at low
   learning rate, and per-member early stopping on an internal 2005 split. Early stopping is
   diagnostic: `best_iteration` hit the cap (799/800) on within-2005 validation while eval AUC got
   *worse* — capacity tuned to 2005 overfits the 2005 → 2006 gap.
2. **Learning the blend** (an XGBoost meta-model stacked over out-of-fold member predictions):
   0.7586 vs 0.7621 for the flat average — the meta-model overfits the OOF prediction distribution.
   Also neutral/negative: bootstrap-style column-and-row subspace diversity per member, depth mixing,
   `min_child_weight`, `reg_lambda`, `gamma`, `max_bin`, `lossguide`/`dart` booster settings.
3. **Extra feature families**: traffic-volume shares, rate-table counts (reliability weights),
   hierarchical smoothing toward the parent key's rate, one-hot airports, weekday/month-interacted
   rate keys, `route×time` (no carrier), day-of-year, deviation-from-hour features, and dropping
   `Distance`/`minute` as "monotone duplicates" (cost 0.0018 — worth keeping both, the two binnings of
   the same variable are not equivalent to a tree). Several of these were first accepted and then
   killed by a **month-based temporal holdout**, which is a much better proxy for the year gap than
   eval-specific gains: `route×time` looked like `+0.010` on eval and collapsed to `0.682` on the
   temporal split.

## What I would try with more budget

The binding constraint is the year gap, not model capacity: within-2005 CV reaches 0.75-0.76 while
2006 eval sits at 0.766, and everything that raises within-year fit beyond a point lowers the
cross-year score. My next moves would be (a) covariate-shift reweighting of the training rows using
only the 2006 *feature* distribution (density-ratio weights; no labels involved) so the trees spend
capacity on the regime the holdout actually occupies; (b) finer "delay cascade" features — the
historical rate of the *preceding* slot for the same carrier/airport pair and rolling airport rates
over the previous 1-3 hours, which capture propagation at a resolution the single-bucket table cannot;
(c) segment-wise blend weights (per time band / airport size class) fitted on out-of-fold member
predictions rather than one global meta-model; and (d) a per-time-band specialist model set with a
shared rate-table backbone. If a second labelled year were available, I would build the rate tables on
the nearest year only and re-tune the smoothing/nfold pair, since the cross-year rate transfer is
clearly the mechanism doing the work.

## Notes

- The inline comment block in `train.py` narrates the finding sequence; the final constants are
  `BUCKET_MIN = 15` and `TE_NFOLD = 2` (the fold comment quotes the earlier 5 → 3 comparison).
- Only `train.py` and this file were written. `run_experiment.sh`, `validate.sh`, `validate.py`,
  `task.json` and `data/` are untouched. `experiments.tsv` records the full 40-experiment trail with
  the keep/revert decision for each.
