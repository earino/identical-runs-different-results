# FINAL — airline dep_delayed_15min (XGBoost, 2005 train / 2006 eval+holdout)

**Best Eval AUC: 0.7588** (experiments #39, commit 1db43eb; validated via `predict_proba` with the
target column removed). Baseline was 0.7141 (30 trees, depth 6, raw columns).

## What mattered most (cumulative +0.045)

1. **Hour-of-day extraction from DepTime** (`hhmm -> hour, minute, minute_of_day` + sin/cos encodings):
   the single dominant delay signal; baseline 0.7141 -> 0.7183 and the substrate everything else built on.
2. **Schedule-structure count features** (fit on train only): log-counts of flights sharing
   Origin×hour, Dest×hour, route, carrier×hour, carrier×origin/dest/route, 15/30-min slots, month cells,
   plus share ratios (e.g. cnt_carrier_origin / cnt_origin, route share) and route median-distance deviation.
   These transfer across years because schedules repeat: 0.7285 -> 0.7351 -> 0.7387, and the
   carrier-footprint family added +0.002 later (0.7567, 0.7571).
3. **Deep trees with early stopping**: max_depth ladder 6 -> 8 -> 10 -> 12 -> 14 -> 16 -> 18 -> 20 -> 24
   (with lr 0.02-0.03, ES on internal folds) was worth ~+0.025 in total; depth 24 at lr 0.03, ~190 rounds.
4. **Aggressive column subsampling**: colsample_bytree 0.75 -> 0.6 (+0.003), 0.55 (+0.001) — decorrelates
   the 4 ensemble members against a wide count-feature matrix.
5. **4-fold bagged ensemble with per-fold early stopping** (predict_proba averages the fold models):
   honest iteration selection, variance reduction, and robustness for the hidden holdout.

## What did not help (reverted)

1. **Target encoding** of carrier/origin/dest/route — both plain smoothed OOF TE and a retry with
   heavy smoothing (m=100) on composite cells (carrier×hour, origin×hour): 2005 delay-rate statistics do
   not transfer to 2006 beyond what the count features already capture.
2. **Calendar/holiday features** (day-of-year, days-to-Jan1/Jul4/Nov11/Dec25, Thanksgiving/Christmas
   windows): no gain at any depth (0.7508 vs 0.7556 when tried), the day-of-month signal is too noisy
   in a 100k-row single-year slice.
3. Micro-tunes that regressed or tied: min_child_weight > 1, colsample_bylevel, per-fold seeds,
   rolling ±1h window counts, share-ratio over-additions, cyclical-feature ablation (they do help),
   lr 0.015 (needs more rounds than the 120 s cap allows), 5-fold / depth-28 (timeouts).

## With more budget

I would first buy runtime headroom (fewer/purer count features, faster `prepare`) to run **5+ folds at
colsample 0.5** — every fold-count and column-sampling step so far moved the eval AUC up, and the two
trends were never tested together due to the 120 s experiment cap. Second, a **two-seed × k-fold
grid** (8+ members) averaged in `predict_proba`, which usually adds another few thousandths of AUC.
Third, I would revisit feature interactions at the deeper tree counts (e.g. origin×hour×carrier
density with hierarchical shrinkage toward origin×hour counts), and try `grow_policy=lossguide`
with a leaf budget, which sometimes finds the asymmetric schedule-block structure that depth-limited
growth misses. None of the delay-rate (TE) directions are worth revisiting: the 2005->2006 shift kills
them, while schedule-density counts keep transferring.
