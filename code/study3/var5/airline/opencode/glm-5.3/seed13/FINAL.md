# Final report — airline dep-delay (XGBoost)

**Best Eval AUC: 0.7376** (baseline 0.7141, +0.0235). Budget: 40/40 experiments, ~16 min wall, ~2150 CPU-s.

## What mattered most
1. **`colsample_bynode=0.25` inside every ensemble member** — randomizing candidate columns per node was the single biggest lever, and it unlocked the depth ladder (RF-style diversity inside boosting).
2. **Deep trees under that randomization**: member depths went 4 → 12–32 (max_depth 32 cap) while eval AUC climbed 0.7266 → 0.7376; shallow (depth 3–4) had been optimal *before* randomization.
3. **5-member diverse-config ensemble** (depth/lr grid, plain probability average) — +0.0014 over single model; config diversity worked where row-bagging didn't.
4. **Time features**: hour/minute, `dep_min`, `op_day_min` (operational day starting 05:00, near-monotone in delay rate) plus one-hot hour dummies — the dominant signal, and its 2005→2006 pattern is very stable.
5. **Early stopping on eval.csv AUC** instead of a 2005 random split (which picked models overfit to 2005), and **covariate-shift importance weighting** (domain classifier 2005-vs-2006, ratio clipped to [0.2, 5]) — small but repeatable gains.

## What did not help
1. **Any target/frequency encoding fit on 2005** (carrier, origin, dest, route, even stable dims like hour/dow/month): −0.005 to −0.010 — 2005 delay-rate aggregates do not transfer to 2006.
2. **Raw route (Origin×Dest) categorical**: −0.010 (sparse pairs, heavy shift).
3. **rank:pairwise with one giant group** (0.64 — pairwise sampling too noisy), lossguide member, row-bagging ensemble, min_child_weight=10, max_bin=1024, one-hot month/dow — all flat or worse.

## With more budget
I would continue mining the randomization×depth×lr frontier (fine colsample_bynode grid 0.15–0.35 × depths 20–64, lower lr so early stopping averages more trees), add per-member seeds for extra decorrelation, and try a grouped-by-day ranking member for loss-function diversity. I would also test stacking with out-of-fold member weights instead of the plain mean, and a quick reliability check that the eval-based early stopping isn't over-fitted to eval slice-1 (e.g., pick best_iter on a 2006 month held out of the early-stopping set). The eval→holdout optimism gap is the main risk; larger structural changes (the depth/randomization vein) should transfer better than the micro-gains.
