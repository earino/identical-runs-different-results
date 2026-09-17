# Final Report — airline delay AUC (XGBoost)

**Best Eval AUC: 0.7379** (baseline 0.7141, +0.0238). Final commit: `289df8f` ("headroom: patience 100, OOF k=3").

## Final architecture
5-member XGBoost ensemble (probability-averaged), trained on 100k rows from 2005, early-stopped on
data/eval.csv (2006 slice 1):
- 3 depth-wise members (depth 10–12, lr 0.05–0.06, mcw 10–20, subsample 0.85–0.9, colsample 0.7–0.85)
  with **feature-view diversity**: full features / no-TE view / drop-interaction-TE view;
- 1 `lossguide` (leaf-wise) member; 1 conservative deep member (depth 14, mcw 50).
Features: raw + cyclical time encodings (hour, minute, sin/cos of time-of-day, month, day-of-month),
slot30/hour categorical, log-distance, weekend/red-eye flags, smoothed target encodings
(carrier, origin, dest, bucket, hour, dow×bucket, carrier×dow, carrier×month, carrier×slot15)
+ log-counts (carrier, origin, dest). Target encodings are fit on train only; training rows use
**out-of-fold** TE values (3-fold) while prediction uses full-train maps.

## Changes that mattered most
1. **Smoothed target-encoding interactions** (carrier×dow, dow×bucket, carrier×month, carrier×slot15,
   bucket, hour): 0.7237 → 0.7320 (+0.008). Time-of-day × carrier × day-of-week is where the signal is.
2. **Ensembling with diversity** (params + feature views + lossguide + conservative member):
   0.7320 → 0.7379 (+0.006). Single tuned model was ~0.724.
3. **Hyperparameters**: depth 10–12, min_child_weight 10–20, lr 0.05, subsample/colsample < 1
   (0.7155 → 0.7237).
4. **OOF target encoding for training rows** + patience 100: small gain to 0.7379 and better
   leak-hygiene for the hidden holdout.
5. Removing the raw `DepTime`-integer outlier handling noise and keeping cyclical encodings
   (hour/minute sin-cos) helped stability.

## Things that did NOT help
- **Route (Origin_Dest) as a categorical or target-encoded feature** — 4.2k levels on 100k rows;
  0.7080 when bundled with route features.
- **Sparse airport interactions** (origin×dow, dest×dow, origin×month, dest×month TEs): 0.7256–0.7316.
- **Bagging on 75% row subsamples** (0.7309) and extra weak members (shallow depth-6: 0.7375;
  no-raw-time view: 0.7346) — diversity gains were already captured.
- Logit- and rank-averaging exactly matched probability averaging (0.7365/0.7379).
- gamma=1.0 (0.7378), TE smoothing 30→100 (0.7343): neutral.

## With more budget
- A small guided search over member configs (depth/mcw/colsample) with 5-fold CV inside train,
  selecting members by OOF AUC rather than eval.csv, then a 8–10 member ensemble;
- DART members with a proper time budget (my single attempt hit the 120 s wall-clock cap);
- per-carrier slot-level interaction TEs beyond slot15 (slot5/minute-level) and a filtered
  route TE only for the top-N most frequent routes.
