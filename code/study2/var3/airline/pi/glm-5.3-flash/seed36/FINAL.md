# Final report — airline delay (XGBoost, autoresearch harness)

**Best Eval AUC: 0.7209** (baseline 0.7141). Final model: equal-weight average of 5 XGBoost
classifiers over the same engineered feature set (depths 2–5, 400–800 trees, lr 0.03–0.05).

## What mattered most

1. **Shallow trees + slow learning.** The path d6@30/lr.1 (0.7141) → d3@400/lr.05 (0.7166) →
   d4–d5@600–800/lr.03 (0.7187). Depth 8 was much worse (0.7095); the sweet spot is depth 3–5
   with many rounds and lr ≤ 0.05. Subsampling/colsample always hurt.
2. **Ensemble of diverse shallow configs** (the single biggest jump, 0.7187 → 0.7204/0.7209):
   averaging depths {2,3,4,5} with matched lr/rounds beat any single model. 8 members was not
   better than 5; per-member seeds changed nothing.
3. **Feature engineering v1** (time-of-day minutes + sin/cos, numeric Month/DayofMonth/DayOfWeek,
   log-distance): −0.002 at d6@30 but +0.0003 once the model regime was fixed — features only pay
   off with enough rounds to use them.
4. **Smoothed target encoding** (Origin/Dest/UniqueCarrier/hour, a=20, fit on train only):
   +0.0004 in the final regime (it *hurt* in the early regime — regime matters).
5. **Holiday-peak flag** (mid-Dec, Jul 1–4, Thanksgiving week, Presidents/Late-May windows):
   +0.0001 — tiny but mechanistically plausible and cheap.

## What did not help

- **Route (Origin_Dest) target encoding** — 0.7066, a big regression; ~16 rows/route is far too
  sparse, and the noisy rate derailed splits.
- **Interaction TEs** (carrier×hour: 0.7152) — 2005 group rates did not transfer to 2006.
- **Capacity/regularization probes** at high lr: 100–150 trees, depth 8, subsample/colsample 0.8–0.9
  (0.7095–0.7125), stronger TE smoothing a=50 (0.7168), hour-categorical instead of raw DepTime
  (0.7115) — the raw hhmm integer already encodes the (nearly monotone) hour→delay curve.

## With more budget

I would (a) tune the ensemble member grid more systematically (weights, per-member colsample for
diversity, a dart-mode member), (b) try OOF (cross-fitted) TE to remove self-leak from the training
matrix, (c) add month×hour and origin×hour interaction *features* (not TE) now that the model has
rounds to exploit them, and (d) verify keep/discard decisions on an internal time-split to reduce
eval.csv selection noise.
