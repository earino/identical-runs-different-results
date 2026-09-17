# Final report — autoresearch XGBoost (airline dep-delay, 2005 -> 2006)

## Result

- Best Eval AUC: **0.7213** (baseline 0.7141, +0.0072)
- Best configuration: 12-member diverse bag on the full feature view
  (raw columns + hour + origin-hour / dest-hour smoothed target encodings) plus 6
  "lean-view" models (numeric-only features, 400 rounds, lr 0.05), all rank-averaged
  (ranks normalized to (0,1), which is AUC-equivalent and contract-valid).
- Commit: 8b96682 + the follow-up contract fix (normalized ranks). `validate.sh` prints
  `CONTRACT OK` and reproduces Eval AUC 0.7213 through `predict_proba`.

## The 3-5 changes that mattered most

1. **Diverse-seed bagging (subsample 0.7, colsample grid, mixed depths 4-10).** Replacing the
   single model with 8->12 decorrelated members was worth roughly +0.002 on its own. Diversity
   across max_depth and colsample_bytree mattered more than adding more same-config seeds
   (16- and 24-member same-recipe bags plateaued).
2. **Origin-hour and dest-hour smoothed target encodings (m=100).** Plain origin/dest/route
   target encodings consistently hurt (their cross-year rate correlation is only ~0.26-0.38),
   but the origin x hour and dest x hour interactions transfer strongly (corr ~0.87-0.89 for
   cells with >=50 training rows). Encoding those two interactions as numerics added ~+0.0006.
3. **Two feature "views" per model family (input-space diversity).** A numeric-only lean view
   (hour, minute, DepTime, calendar numerics, log-distance, dep sin/cos, 400 rounds) trained
   separately from the categorical-heavy full view and rank-averaged in gave the single
   biggest jump of the session: 0.7177 -> 0.7208 (+0.003). Errors of the two views are
   genuinely decorrelated; this beat every attempt to enrich a single view.
4. **Rank averaging instead of probability averaging.** AUC only cares about ordering;
   averaging normalized ranks made members contribute equally regardless of their raw
   score scales. Small but repeatable gain (+0.0001) and free.
5. **Keeping capacity low on the categorical view (60 rounds, depth<=10).** Every attempt to
   scale up single-model capacity overfit the 2005->2006 shift (0.7056-0.7083 in early runs);
   the bag + low-capacity members is what generalized.

## Things that did not help

- **More single-model capacity** (2000 rounds w/ early stopping, depth 8-10): always worse
  than the 30-tree baseline on the time-shifted eval.
- **Carrier / origin / dest / route / hour-alone target encodings**: route TE and carrier TE
  repeatedly scored worse (route rates barely transfer year-over-year; carrier is redundant
  with its categorical column).
- **Feature clutter**: numeric month/dow and their interactions with hour, distance
  interactions (log-dist, dist x hour, dist-vs-route-mean), sin/cos on the *full* view,
  route as an explicit categorical, 30-minute TE slots, hour monotone constraint — all flat
  or slightly negative. The model already extracts these from the raw columns.

## What I would try with more budget

The clear structural winner was input-view diversity, so I would push that harder: a third
view made of only the smoothed target encodings plus numerics (no raw categoricals), a
Carrier x hour TE view, and possibly K-fitted target encodings (out-of-fold TE to remove the
self-leakage in the fitted encodings), then weight views by out-of-fold AUC rather than
uniform rank averaging. Second: the lean-view members improved monotonically with more
rounds (200 -> 400), suggesting they are still underfit; I would try 800 rounds at lr 0.03
with early stopping on a holdout, and give the lean view 2-3x the vote weight. Third: with
1M-row holdout in mind, I would sanity-check the top-3 configs on a simulated year split
(train on Jan-Oct 2005, validate Nov-Dec 2005) to pick the most time-robust variant, since
eval.csv (2006 slice 1) and the hidden holdout (2006 slice 2) may drift differently.

## Experiment log summary (experiments.tsv)

| # | change | AUC |
|---|--------|-----|
| 1 | baseline (30x6 cats) | 0.7141 |
| 11 | 8-bag subsample .7/colsample .8 | 0.7161 |
| 14 | + hour feature | 0.7162 |
| 16 | bag 60 rounds | 0.7164 |
| 20 | + OH/DH TE m=100 | 0.7170 |
| 24 | diverse depths bag | 0.7176 |
| 27 | rank averaging | 0.7178 |
| 31 | + lean-view ensemble | 0.7208 |
| 34 | lean 400 rounds | 0.7212 |
| 39 | 12 full + 6 lean | **0.7213** |
