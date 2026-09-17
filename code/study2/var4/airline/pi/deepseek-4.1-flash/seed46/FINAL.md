# Final Report — airline `dep_delayed_15min`

**Best Eval AUC: 0.7393** (experiment #38, commit `a504f8d`, "model: 40-member ensemble").
Contract validation passes: `./validate.sh` → `CONTRACT OK`, eval AUC 0.7393 via `predict_proba(df)`
with the target column removed. Baseline was 0.7141.

The final `train.py` is an ensemble of **40 diverse `XGBClassifier` members**
(`tree_method="hist"`, `enable_categorical=True`, `learning_rate=0.10`, varying depth 4–9,
subsample, colsample, `reg_lambda`, `min_child_weight`; seeds `SEED+i`). One early-stopping probe
on a 15% internal split (never eval) fixes a shared tree budget used by every member, so each member
is fit exactly once on the full training set. All feature engineering lives inside `prepare(df)` and
every encoder/statistic is fit on training data only.

## Changes that mattered most

1. **Train-fitted congestion / frequency encodings** (origin, destination, carrier) at hour, 30-min and
   15-min granularity, plus fractional shares and peak-hour ratios. This was the first large, robust
   gain and introduced the "airport busyness" signal that the rest of the work built on.
2. **Lead/lag congestion windows** — counts at the same airport in the previous/next hour and previous/
   next 30-minute bin. This was the single biggest jump (series of +0.002…+0.0035 steps) and is a genuine
   delay-propagation signal: a flight is more likely delayed when its airport was/is congested around it.
3. **Carrier hub size and hub share** (`carrier_origin_freq`, `carrier_dest_freq`, and their shares of the
   airport's traffic), which capture carrier-specific hubbing effects.
4. **Large diverse XGBoost ensemble with a shared tree budget.** Scaling 14 → 24 → 32 → 40 members gave a
   steady monotone gain, and moving to `learning_rate=0.10` made each member cheap enough to afford the
   larger ensemble within the 120 s budget.
5. **Native categorical handling** for the low-cardinality string columns, with high-cardinality columns
   (e.g. routes) deliberately excluded.

## Things that did not help

1. **High-cardinality route/OD features** (raw `Origin_Dest` categorical, route frequency, route-hour,
   route competition). Route identity does not transfer across years and consistently hurt.
2. **Finer/coarser lag granularity beyond prev/next hour + 30 min** — 15-minute lead/lag and 2-hour lag
   gave no gain (and pushed runtime into timeouts).
3. **XGBoost tuning knobs in isolation**: `max_bin=128`, `max_cat_threshold=128`, `lossguide` growth, and
   per-member learning-rate round scaling. Also unhelpful: daily-volume counts, seasonal shares
   (origin/dest × month/weekday), and deeper single trees.

## What I would try with more budget

The dominant signal is temporal congestion around a flight, so the next step would be to make it
spatio-temporal and graph-like: model upstream/downstream airport congestion (e.g. delay state at the
flight's origin in the preceding block plus the delay state of the aircraft's likely previous
destination), which requires a route/aircraft-flow graph rather than per-airport counts. In parallel I
would replace the native categorical handling with properly nested out-of-fold target encoding for
origin/dest/carrier, add external weather proxies (month × origin is a weak proxy at best), and use a
small stacking/weighted-blend layer over the ensemble instead of a plain mean. Finally, given the run
was CPU-bound, feature selection on the redundant frequency family would free compute for an even larger
ensemble, which showed a reliable (if diminishing) return.
