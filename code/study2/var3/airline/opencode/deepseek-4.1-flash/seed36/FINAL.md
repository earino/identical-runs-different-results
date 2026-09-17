# Airline departure-delay prediction — final report

**Best Eval AUC: 0.7351** (baseline 0.7141, +0.0210). Commit `795699f`.
Validated with `./validate.sh` -> `CONTRACT OK`, eval AUC via `predict_proba` = 0.7351.

## Setup

`train.py` builds all features inside `prepare(df)` so the same code path runs on the
hidden holdout. Target encodings are fit on `data/train.csv` only; for training rows they
are computed out-of-fold (5-fold), for inference with the full-train maps.

Final model: an average of 32 XGBoost classifiers (depths 3–10 × 4 seeds, mixed
`depthwise`/`lossguide`, randomized subsample/colsample/min_child_weight), `n_estimators=220`,
`learning_rate=0.05`, `max_bin=512`, `enable_categorical=True`.

## Changes that mattered most

1. **10-minute departure-slot target encodings (biggest win, +0.010).** The exact scheduled
   departure time is the dominant signal. Encoding the 10-minute bin (`DepTime//10*10`) as a
   smoothed target rate, plus its interactions with `Origin`, `Dest`, `UniqueCarrier`,
   distance-bin, `DayOfWeek` and `Month`, gave 0.7234 -> 0.7336. These interactions are very
   stable between 2005 and 2006 (per-group target-rate correlations 0.83–0.99, vs 0.38–0.59
   for raw airport effects), so they transfer to the held-out year.
2. **Hour-level interaction target encodings.** `Origin×hour`, `Dest×hour`, `Carrier×hour`,
   distance-bin×hour, with smoothing 20. First structural gain (+0.0033).
3. **Ensemble averaging.** 32 diverse trees averaged on probability: 0.714 -> 0.717 alone and
   it makes the target-encoded features pay off (0.719 -> 0.735). Mixing `lossguide` and
   `depthwise` added +0.0007.
4. **Residual features** `te_{origin,dest,carrier,db}_t10 - te_t10` (airport/carrier effect
   relative to the time-of-day baseline).
5. **Smoothing / capacity tuning**: `SMOOTH=20` (vs 50/150/10), `max_bin=512`, `n=220, lr=0.05`.

## What did not help

- Naive target encoding of high-cardinality keys (route, Origin, Dest) and adding route as a
  raw categorical — both overfit the 2005 slice (0.704–0.710).
- Frequency/count encodings (<= 0.0001), exact-minute (`t1`) encodings, coarse 20/30-minute
  encodings, month/dow×hour keys added on top of the hour keys.
- Rank averaging across ensemble members, dropping raw `DepTime`, DART boosting and
  `max_bin=1024` (slower, no gain / timeouts), random `reg_lambda`/`gamma`.

## With more budget

Try an XGBoost stacker trained on out-of-fold predictions of the ensemble members (the
contract allows XGBoost only, so a logistic meta-learner is out). Also worth: per-key
smoothing strengths, calibration of the individual members before averaging, and an
explicit time-series split for early stopping to better match the 2006 holdout.
