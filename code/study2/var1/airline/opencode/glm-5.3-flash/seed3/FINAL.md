# Final report

**Best Eval AUC: 0.7305** (baseline 0.7141, +0.0164). Contract validated (`CONTRACT OK`).
All 40 experiments used; HEAD = the best `train.py`.

## Final model

XGBoost ensemble of **4 hist-tree models** (depth 10, lr 0.05, 1500 trees, mcw 20, lambda 1.0,
subsample/colsample jitter, different seeds), probability-averaged. The key structural ingredient:
each member is **blinded to a different feature slice** (member-level feature dropout):

- M1: all 18 features
- M2: no Distance/LogDist, no DepTime_cos
- M3: no calendar (Month/Day/DoW), no Route pair (keeps Origin/Dest, time, distance)
- M4: no UniqueCarrier, no Dest

Features: Month/DayofMonth/DayOfWeek as ints, DepTime raw + mod-2400 + sin/cos of minute-of-day +
DepHour categorical, Distance + log1p, native categoricals (carrier, origin, dest, route, hour).

## Changes that mattered most

1. **Engineered time features + native categoricals incl. Route** (0.7141 → 0.7183): int-parsed dates,
   DepTime mod/sin/cos + hour cat, Origin→Dest Route as one categorical, bigger model
   (1500 trees, lr .05, depth 10) with early-stop-then-full-retrain.
2. **3-member seed/jitter ensemble** (0.7183 → 0.7213): simple averaging of jittered configs beat any
   single model; 4th plain member added nothing.
3. **Member-level feature dropout** (0.7213 → 0.7237 → 0.7243 → **0.7290**): forcing members to learn from
   different feature slices (calendar-blind, distance-blind, and finally also Route-blind M3) decorrelates
   errors far more than seed jitter; the Route-blind M3 alone was worth +0.0047.
4. **Carrier/Dest-blind 4th member** (0.7290 → 0.7305): a fourth differently-blinded view kept adding.

## What did not help (all reverted)

- **Target encoding** (full-train: 0.6891 — leakage collapsed early stopping to 72 trees; honest OOF 5-fold:
  0.7138) and **count encodings** (0.7171): native categoricals already capture location signal better.
- **Interaction categoricals** Origin×Hour / Carrier×Hour (0.7154); **canonical route + direction flag** (0.7122);
  **date fields as categorical** (0.7057).
- **Depth sweep**: depth 8 (0.6895, early stopping fell into a val-curve dip), depth 12 (0.7014); lossguide
  (timeout); DART member (timeout). More trees / lower lr (3000 @ lr .04: 0.7182): plateau at ~1500.

## With more budget

I would push the dropout-ensemble further: more members blinded to other slices (e.g., no-DepHour,
no-Origin, carrier-only), per-member tree counts, and a small greedy weight search on a time-shifted
validation scheme. I would also revisit OOF target encoding *inside* ablated members (e.g., TE for
Origin/Dest in the route-blind member only), since blinded members have capacity to exploit weaker
signals that the full model absorbs.
