# Final report — airline dep-delay AUC (autoresearch XGBoost)

**Best Eval AUC: 0.7475** (baseline 0.7141 → +0.0334). Final `train.py` = commit `5e4703b`
(experiment 8), validated with `./validate.sh` → `CONTRACT OK`.

Final model: 3-model XGBoost ensemble, each `n_estimators=600, lr=0.05, max_depth=16`,
structurally diverse (`max_bin=1024`, `max_bin=2048`, `max_bin=1024 + colsample_bynode=0.7`),
trained on the full 100k train.csv with linear month-recency sample weights (Dec = 3× Jan),
simple mean of probabilities. Features: `DepTime`, `hour = DepTime//100`, `Distance`,
`UniqueCarrier`/`Origin`/`Dest` as native categoricals — nothing else survived the 2005→2006 shift.

## Changes that mattered most
1. **Train on the full 100k at high capacity, no early stopping** — ES on a 2005 holdout stopped
   far too early (~120-470 rounds); refitting on all rows with fixed deep configs jumped
   0.7141 → 0.7305 → 0.7427 (d9→d14, n=800).
2. **Fine histogram bins + hour feature** — `max_bin=1024` (and 2048 for one member) plus
   `hour = DepTime//100`: 0.7427 → 0.7453.
3. **Structural 3-model ensemble** — depthwise d16 members with bin-width and
   colsample_bynode diversity, probability-averaged: 0.7453 → 0.7464.
4. **Month-recency sample weighting** (α=2 linear, i.e. weight = 1 + 2·(month−1)/11) — exploits
   the time-shifted eval year; lifted every member and the ensemble: 0.7464 → 0.7470.
5. **Member swap to d16 family + csn-diverse member** (d16b1024 / d16b2048 / d16b1024+csn0.7):
   0.7470 → 0.7475.

## Things that did NOT help (all reverted)
1. **Every added feature** — route pairs (native categorical or target-encoded), date features
   (Month/DayofMonth/DayOfWeek as int, categorical, or cyclic sin/cos), target/frequency
   encodings, origin×hour interaction TEs, route TEs. Each hurt eval 2006 AUC by 0.001–0.02:
   they memorize 2005-specific patterns that do not transfer.
2. **Early stopping on a 2005 split, lossguide and dart members** — ES lost ~0.008 vs fixed-n
   full training; lossguide (121s+/member) and dart (0.740, 292s) were both infeasible or worse
   inside the 120s cap.
3. **Regularization/fine-tuning** — subsample, colsample_bytree, min_child_weight, gamma,
   lambda, lower lr+more rounds: all neutral-to-worse. Fine-tuning the booster on Oct–Dec 2005
   only collapsed to 0.729 (catastrophic drift on a small subset).

## With more budget I would try
More ensemble members were the clearest untapped gain: offline, 4–5 diverse weighted members
reached 0.7476–0.7477 but needed 130–165s, over the 120s per-run cap — with a higher cap (or
cheaper members: n=400–450, QuantileDMatrix reuse, shared binning) that is directly reachable.
I would also build a proper time-shift validation (train on Jan–Sep 2005, validate on Oct–Dec 2005)
to tune the recency weighting and n without touching eval.csv, and try bagged fold-ensembles at
full n (needs ~200s/run). The single biggest known lever for this dataset, though, is more
training rows; with only 100k, capacity + variance reduction is where the remaining juice was.
