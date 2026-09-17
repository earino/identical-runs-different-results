# FINAL REPORT — Airline Delay AUC Maximization

## Result

| Metric | Value |
|---|---|
| **Final eval AUC** | **0.7458** |
| Baseline (exp 1) | 0.7141 |
| Improvement | **+0.0317** (+4.4% relative) |
| Experiments used | 40 / 40 |
| Wall clock | ~53 min of 230 (all 40 done with 177 min to spare) |
| Python CPU | 9,097 s of 18,000 |
| Contract check | `validate.sh` → **CONTRACT OK** (0.7458 via `predict_proba` with target column removed) |

Final model (`train.py`, commit `1a025db`): equal-weight mean of **23 XGBoost members**, all sharing one
recipe — depth 6, eta 0.05, min_child_weight 20, **colsample_bynode 0.3**, hist tree method with
`enable_categorical=True` — differing only in which year-stable DepTime features and interactions they
receive, and in boosting rounds (400–1500). `predict_proba(df)` re-runs the full feature pipeline in
`prepare(df)` (encoders/categories fitted on train only), builds one DMatrix, and averages the 23
booster outputs. Runtime 116 s (< 120 s limit), 4 threads.

## Key findings (what actually transferred 2005 → 2006)

1. **Year drift dominates everything.** Target encodings, route features, airport-level statistics, and
   any Origin/Dest categorical interaction consistently *hurt* eval AUC while helping 2005 CV. The target
   rate by hour of day (0.04 → 1.0) is stable across years, but airport/carrier-specific rates drift.
   Only year-stable structure transfers: time-of-day, calendar, carrier, and *interactions of carrier
   with time* (not with airports).

2. **The scheduled-departure-time axis was the big unlock (exp 23–25).** Adding an hour-of-day categorical
   (`hour_cat`) plus **multi-resolution DepTime bins** (10/15/20/30/45/60 min) to the champion recipe took
   solo AUC from 0.722 → 0.730, and 15+30-min bins to 0.733. This single discovery drove
   0.726 → 0.746.

3. **Carrier × DepTime-bin interactions generalize (exp 27–39).** `carrier × block15/20/30/45` pushed
   solos to ~0.736–0.739. Origin × bin interactions help modestly (0.733–0.734); dest × bins and
   dow × bins do not. Longer rounds on interaction members help up to ~1000–1500 trees.

4. **Simple equal-weight means of diverse members beat everything else.** OOF stacking with gbtree/gblinear
   meta-learners (exp 17), rank-averaging, weight tuning, and seed bagging (exp 34) were all tested and
   rejected. More good members ≈ monotonically better: going from 7 → 23 members gave
   0.7339 → 0.7458.

5. **Run-to-run noise is ~±0.001**, so only structural changes (new feature views, new interaction
   families) were trusted over noise-level tweaks.

## Progression

```
exp 1   0.7141  baseline (30 trees, depth 6)
exp 7   0.7200  small/shallow members transfer best; first mean-ensemble
exp 15  0.7261  colsample_bynode 0.3 = best solo recipe (0.7220)
exp 22  0.7267  mean-ensemble plateau
exp 23  0.7287  hour_cat + block30 discovered (solo 0.7301)
exp 25  0.7339  block-bin family mean, legacy members dropped
exp 30  0.7395  carrier×time interactions
exp 33  0.7437  interaction zoo (origin/dest/carrier × bins)
exp 38  0.7455  long-round interaction members (800–1500 trees)
exp 40  0.7458  final locked 23-member ensemble
```

## Rejected directions (tested, did not beat the final mean)

- Target encoding (any smoothing), route features, count features — year drift.
- OOF stacking (gbtree + gblinear meta) — 0.7259 vs 0.7260 simple mean (exp 17).
- Rank-averaging, member weights, seed bagging, dart boosters (timeout), deeper solo models
  (depth 8–10 with legacy features), minute-of-hour, week-bin, carrier×origin, dow×bin interactions.

## Reproduce

```bash
./validate.sh   # trains, evaluates, checks predict_proba contract → Eval AUC: 0.7458
```
