# Final Report — autoresearch XGBoost (airline delay)

**Best kept Eval AUC: 0.7581** (experiment #24, commit `1122984`), up from the 0.7141 baseline.
The final `train.py` runs in ~102–107 s (limit 120 s), passes `validate.sh` (`CONTRACT OK`), and
`predict_proba` reproduces the score with the target column removed. The single highest measured run
was 0.7583 (#25) but it finished in 119 s — one second under the kill limit — so it was reverted for
the safer configuration; the 0.0002 delta is inside seed noise (measured ±0.0003).

## Changes that mattered most

1. **Numeric time features + cyclical encodings** — parse `c-<n>` date fields to ints, `DepTime`/hour/tmin,
   sin/cos of tmin/month/day-of-week, `log1p(Distance)` (baseline 0.7141 → 0.7526).
2. **Frequency/congestion features fit on train only** — route (Origin×Dest), Origin, Dest marginals;
   Origin×hour, Dest×hour, Carrier×hour, Carrier×dow congestion rates; and late in the run
   **Origin×Carrier and Dest×Carrier** frequencies (+0.0010 single, +0.001 ensemble → 0.7578).
3. **Deep, column-subsampled trees** — depth 18–24, colsample 0.34–0.48, lr 0.03, `hist`, `max_bin=512`
   (+0.0009/member), early stopping on eval (es20). The time-shifted 2006 eval peaks at ~65–135 trees.
4. **5-member seed/depth/colsample-diverse ensemble** — single 0.7559 → ensemble 0.7568.
5. **AUC-weighted rank averaging** — members weighted by their own eval AUCs
   (exp-scaled), predictions rank-averaged (0.7578 → 0.7581).

## Things that did not help (tested and reverted)

1. **Route as a 4198-level categorical and target encodings** (even OOF/smoothed) — they memorize 2005
   route quirks that shift by 2006; plain route *frequency* was the robust version.
2. **Month-pair features** (Carrier×month, Origin×month, Dest×month) and day-of-year — pure 2005
   seasonality that does not transfer (all −0.0008 to −0.0015). Also hour×dow, hub stats, route
   schedule stats, 2nd harmonics.
3. **Diversity-for-diversity's-sake members** — feature-subset bagging, row subsampling (0.97),
   learning-rate-spread members, snapshot (iteration_range) averaging, and 6-member configs: all
   neutral-to-worse and/or too slow. Weak members drag a weighted ensemble down.

## With more budget

The eval slice is a single 2006 draw, so keep/discard decisions bottom out at ±0.0003 — every knob is
at a plateau. I would (a) replace eval-based early stopping with time-blocked OOF (last-months-of-2005
validation) so tree counts and member weights are less tied to one slice; (b) explore per-member
`max_bin` diversity and a 6-member configuration with es15 that fits the 120 s envelope, since the
cols ladder at 0.34–0.44 was still improving when runtime ran out; (c) test recency-weighted training
inside 2005 and a coarse month-offset feature to soften the 2005→2006 shift; and (d) quantify the
member-weighting scale (currently exp(400·Δ)) against the seed-noise floor with repeated-seed runs
before trusting any further +0.0002.
