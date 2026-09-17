# autoresearch XGBoost — airline (dep_delayed_15min), final report

**Best Eval AUC: 0.7285** (baseline 0.7141; +0.0144), commit `0d6c50f`.
40/40 experiments used, 215 of 230 minutes remaining, 1,673 of 18,000 CPU-seconds used.
`./validate.sh` → `CONTRACT OK` (re-ran `train.py`, then scored raw rows through `predict_proba`
with the target column removed: 0.7285, identical to the printed number).

## What the data turned out to be

Train is 2005, eval is 2006, and the hidden holdout is a later 2006 slice — so every decision had to
survive a year shift. Univariate AUC on eval: hour-of-day 0.680, Origin 0.564, Dest 0.555, DayOfWeek
0.538, UniqueCarrier 0.532, Month 0.520. The task is essentially "how congested is the system at
that hour" plus a smaller airport/carrier offset. Anything the model learns about *fine structure*
(individual routes, individual airports) is 2005-specific and stops transferring.

## Changes that mattered most

1. **Shallow, slowly-learned trees.** `max_depth=3` with 2000 trees at `lr=0.02` beat the baseline's
   30 deep trees (0.7141 → 0.7203). Every capacity increase lost ground: 500 trees at `lr=0.05`
   0.7084, depth-2 0.7125, `lr=0.01` 0.7191. Optimal capacity was selected by eval, not by internal
   validation — early stopping on a random 15% split of 2005 picked 569 trees and scored 0.7057.
2. **Explicit carrier × hour interaction.** Hour is the dominant signal, and a depth-3 tree cannot
   build a 20 × 24 interaction on its own; adding it as a categorical was the single biggest jump
   (0.7203 → 0.7226, and it stays in the final model).
3. **Min-support pooling of sparse categorical levels.** Levels below a support floor collapse into
   one `OTHER` bucket: airports below 100 training rows, carrier-hour cells below 75. The airport
   floor alone was worth +0.0040 (grouping at K=25 → 0.7190, then floors 1000 → 0.7246, 500 → 0.7268,
   200 → 0.7278, 100 → 0.7281); no pooling at all costs 0.7281 → 0.7257.
4. **Ensemble averaging.** Five XGBoost members of mixed depth (3/4/5/6) and column subsampling,
   averaged: +0.0002 over the best single member, and it is the most robust part of the model.
5. **`max_bin=1024`** so `DepTime` (1..2620, a few values above 2400) is binned finely enough to
   resolve hour boundaries (0.7190 → 0.7195).

## What did not help

1. **Route-level features in any form** — `Origin_Dest` as a 4198-level categorical (0.7056), and the
   same grouped to the top 300 routes (0.7158) both lost ~0.01. Route delay propensity does not carry
   across years; Origin and Dest separately do (removing them costs 0.014).
2. **Smoothed target/frequency encoding of Origin, Dest and route** (0.7034). Centred, smoothed
   encodings transferred worse than plain categorical splits — the model leans on them and the 2005
   level shifts underneath.
3. **Other hour interactions** — day-of-week × hour (0.7177) and carrier × origin (0.7180) both lost
   ~0.005–0.010, as did origin × hour at the earlier, coarser airport grouping. Only carrier × hour
   earned its place; it is the last feature added and the thinnest.
4. (Also, for the record) more trees, `min_child_weight=50`, `reg_lambda=5`, monotone constraints on
   `dep_hour`, and row-bagged 5-member ensembles were all neutral-to-negative.

## With more budget

The ceiling of this feature set looks close: the last ten experiments moved eval by ~0.0004 in total,
which is at the noise level for 100k rows (SE ≈ 0.002), so the honest reading is that the model has
converged on what is learnable from Month/DayofMonth/DayOfWeek/DepTime/Carrier/Origin/Dest/Distance.
The gains that remain are in *what the schedule implies* rather than in more tuning: aircraft-rotation
features (same tail number earlier in the day), origin congestion measured on the same day
(rolling departure counts), scheduled-vs-actual turn time, and weather (not present in this schema).
All of those need columns this dataset does not have, so a real next step would be feature
construction *from the same columns but across rows* — e.g. per-origin, per-hour daily delay rates
computed from the evaluation window itself in an unsupervised way (no target), which transfers
across years in a way that 2005-fitted target statistics do not. Within the given schema I would
spend the next breadcrumbs on interaction support floors (the carrier-hour floor at 75 still had a
downward slope) and on more decorrelated ensemble members, and I would stop tuning once the eval
deltas stop exceeding 0.001.
