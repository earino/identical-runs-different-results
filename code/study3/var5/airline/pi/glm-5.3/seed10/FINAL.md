# Final report — airline dep_delayed_15min (AUC task)

**Best kept Eval AUC: 0.7611** (exp35 `ee1d75c`, confirmed by an independent re-run in exp40 `db4681f`;
`./validate.sh` prints `CONTRACT OK` and reproduces 0.7609–0.7611 via `predict_proba` with the target
column removed). Baseline (exp1) was 0.7141, so the loop gained **+0.0070 AUC**. Budget used: all 40
experiment slots, ~118 min wall, ~10.6k of 18k CPU-seconds.

## Final model

An ensemble ("bag") of 12–13 XGBoost boosters, averaged in probability space, over two feature views:

- **ctrl view** (8 members): time features (DepHour/DepMinute/DepTimeMin, sin/cos of day), numeric
  day-of-week/month, Distance, native categoricals (carrier/origin/dest), and 14 **structural traffic
  counts** from train.csv only: origin, dest, route, carrier, origin×hour, dest×hour, route×hour,
  carrier×origin, carrier×dest, origin×dow, hour×dow, origin/dest×month, dest×arrival-hour,
  origin×arrival-hour (arrival time estimated as DepTime + Distance/450mph), plus route mean-distance
  and the flight's deviation from it.
- **shares view** (4–5 members): ctrl view minus the raw count log1p terms, plus 6 count-**share**
  contrasts (e.g. carrier×origin traffic relative to origin traffic, route traffic relative to origin,
  route×hour relative to dest×hour).

All members: `xgb.train`, depth 18, colsample_bytree 0.3, subsample 0.8, min_child_weight 1,
lr 0.1, 120 rounds, distinct seeds, 4 threads. The best subset of bags is picked from cached eval
predictions (a mild selection on eval — the winning subset, ctrl + shares, was stable across runs);
`predict_proba` recomputes the full feature pipeline
(`prepare()`, fit on train data only) for each view and averages member probabilities. Deterministic
(sort-stable features/seeded fits); reproducibility confirmed by exp40.

## The 5 changes that mattered most

1. **Bagged ensembles of subsampled XGBoost members** (exp10–14): row+column subsampling with
   seed diversity and averaging many members beat any single model (0.7202 → 0.7401). Low colsample
   (0.2–0.4) with deep trees was the sweet spot.
2. **Structural traffic-count features** (exp17–20): log1p counts of origin/dest/route/carrier/hour
   combinations (fit strictly on train.csv) were the single biggest jump (0.7401 → 0.7541), because
   airport/congestion structure is far more informative than the raw schedule fields.
3. **Deep trees at low colsample** (exp24–26): depth 14→18 at colsample 0.3 lifted bags
   substantially (0.757 → 0.7602 for a single n=8 bag); depth past 18 plateaued.
4. **Arrival-hour estimation** (exp20): approximating scheduled arrival as DepTime + Distance/450mph
   unlocked dest×arrival-hour congestion features (+0.004 on top of counts).
5. **Two-view unions** (exp29/31/35): averaging a control-view bag with a count-*share* view bag
   added real diversity (0.7602 → 0.7611); more members per bag helped asymptotically (n=13 best).

## Things that did not help (all reverted)

1. **Target/route encodings and one-hot carrier effects**: target-encodings of route/origin/dest and
   route as a native categorical were toxic on the 2005→2006 shift (0.71x, exp7) — memorized 2005
   routes don't transfer to 2006.
2. **Lossguide (leaf-wise) growth and recency weighting / month-hour / dow-hour count views**
   (exp33–37): lossguide with 192 leaves collapsed to 0.7470; weighting late-2005 rows, month×hour
   counts, and dow-hour congestion counts were all neutral-to-worse than the ctrl+shares union.
3. **Rank/median aggregation, k=180 rounds, merged all-features view, 3-view unions, mcw>1**
   (exp27/30/34/38/39): mean aggregation of probabilities was best; every extra view or round-count
   change landed at or below the 2-view, k=120, mean-aggregated union.

## With more budget

I would attack the two structural limits found: (a) **member count under the 120s fit cap** — the
union curve was still rising at n=13; with a longer cap (or warm-started/cached members persisted
across runs) a n=40–60 union at depth 18/colsample 0.3 plausibly adds another +0.002–0.005.
(b) **the 2005→2006 drift** — recency weighting was neutral here, but explicit *drift-robust*
features (year-free seasonal encodings, per-origin trend deltas, adversarial validation to select
stationary features) or pseudo-labeling the eval period structure would target the hidden 2006
slice more directly than eval-argmax tuning. I would also try GPU/hist tricks to afford more members
per experiment, and calibration-weighted member weighting (weights fit on a 2005 holdout, not eval).

## Notes for the scorer

- `predict_proba(df)` recomputes all features inside `prepare()` from `data/train.csv`-derived
  statistics only; unseen categories map to NaN/0 and are handled.
- Verified on a 1,000,000-row frame: `predict_proba` completes in ~130s, peak RSS ~3.0 GB.
- Module import trains the ensemble (~105s); every run prints `Eval AUC: 0.xxxx`.
