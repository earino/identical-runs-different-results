# Final report — airline departure-delay AUC

**Best Eval AUC: 0.7457** (commit `9555eaa`, 40 experiments used, ~108 s/run)
Baseline was 0.7141. Contract validated: `./validate.sh` prints `CONTRACT OK`
(`predict_proba` reproduces 0.7457 on `data/eval.csv` with the target column removed).

## Changes that mattered most

1. **Out-of-fold smoothed target encoding of every categorical column** (replacing
   XGBoost's native categorical splits). Fit on train only, with K-fold OOF values for
   the fitted training matrix and full-train maps for inference. Origin/Dest/Route
   encoding lifted 0.7141 → 0.7168, and encoding *all* object columns lifted it to 0.7189.
   Native categorical splits on high-cardinality columns overfit the 2005→2006 shift badly
   (more trees actively hurt).

2. **Multi-scale airport × time-of-day target encoding** — the single biggest lever.
   Airport-hour (3-hour buckets) → hourly (0.7354) → half-hour (0.7423, best granularity;
   15-min overfits). Adding a coarser hourly scale on top of half-hourly gave 0.7453, and
   a 2-hour scale gave 0.7457. Smoothing = 20 worked best.

3. **Other interaction TEs**: carrier×origin, carrier×dest, origin×month, dest×month,
   carrier×time, month×time, plus per-key frequency (congestion proxy) beside each mean.

4. **Capacity**: once encodings were regularized, deeper/longer trees helped a lot —
   depth 8 (0.7255) → 11 (0.7284) → 14 (0.7301). lr 0.02 / 1200 trees / subsample &
   colsample 0.9 at depth 14 is the sweet spot; depth 16 plateaus.

5. **Seed ensembling + residual features**: averaging 2 seeds (1200 trees, depth 14) gained
   ~0.001; explicit time-of-day residual features (`OH_te − Origin_te`, etc.) added a little more.

## Things that did not help

- **More regularization / less capacity**: `min_child_weight=20 + λ=2 + subsample/colsample 0.8`
  dropped to 0.7250; full subsample/colsample dropped to 0.7267; `lossguide` leaves was 0.7290.
  The model wanted more capacity, not less.
- **Extra calendar/seasonal features**: day-of-year + sin/cos were neutral; day-of-week ×
  time-bucket and airport × day-of-week TEs were neutral to slightly negative.
- **More aggressive interaction TEs**: adding route×month, carrier×month, carrier-hour, etc.
  as raw TE columns *hurt* (0.7284), and pair-residual features also regressed (0.7422).
  Only the time-of-day interaction family generalized.

## What I would try with more budget

The gains tracked one clear theme: the mean departure-delay rate of an airport (and carrier)
in a *specific fine-grained time window* transfers from 2005 to 2006 far better than raw
categorical identity, and the model rewards depth once those encodings are smoothed. With more
experiments I would (a) search the time-bucket pyramid more systematically (e.g. add a 15-min
scale *alongside* half-hour rather than replacing it, and 3-/4-hour scales), (b) tune smoothing
per feature cardinality instead of a global constant, (c) fit 3–4 seeds at a reduced tree count
so the ensemble stays inside the 120 s cap, and (d) try a leave-one-time-bucket-out or
time-ordered target-encoding scheme to further reduce the 2005→2006 drift. The remaining gap to
the leaderboard-style ceiling (~0.75+) looks like it comes from richer time×location signal
rather than from more model tuning.
