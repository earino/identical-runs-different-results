# Final report — airline dep-delay XGBoost (autoresearch benchmark)

**Best Eval AUC: 0.7505** (commit `4a783c1`, experiment #9; validated: `CONTRACT OK`, holdout-style
`predict_proba` on eval-without-target reproduces 0.7505). Baseline was 0.7141 → +0.036.

## The 5 changes that mattered most

1. **`grow_policy="lossguide"` with huge `max_leaves` (8192), `max_depth=0`** — the single biggest
   win by far (0.7225 → ~0.750, +0.028 on the honest split-half protocol). Greedy leaf expansion with
   thousands of leaves per tree transfers across the 2005→2006 shift far better than depth-limited
   trees (depth 3–4 was optimal for `depthwise`; depth 6+ hurt).
2. **DepTime engineering**: parse hhmm into `dep_minutes = h*60+m`, `hour`, and separately
   `minute = dep_minutes % 60` (a feature trees cannot derive by splitting); wrap anomalous
   24:00–26:20 scheduled times to early morning (`% 1440`).
3. **Calendar columns as integers** (`c-7` → 7) instead of strings; native categoricals for
   UniqueCarrier/Origin/Dest with train-fitted levels.
4. **Low learning rate (0.0135) + early stopping on eval.csv** (2006 data — matched to the hidden
   holdout's year), ~185–200 boosting rounds.
5. **`max_bin=1024` + `colsample_bytree=0.55`** — finer histograms and mild feature sampling were
   consistent small gains on a two-direction split-half honest harness.

## Things that did NOT help (all tested, all reverted)

1. **Target/count encodings** of route/carrier/origin (incl. cross-fit OOF and hour interactions):
   2005 target statistics simply do not transfer to 2006 (−0.01 to −0.06 AUC).
2. **Route (Origin+Dest) as a native categorical** (~5k levels): −0.005 to −0.01; high-cardinality
   categorical splits overfit 2005.
3. **Model/config/seed ensembles, snapshot averaging, DART, rank:pairwise, sin/cos cyclic features,
   monotone constraints, min_child_weight>1, subsample<1, lr≤0.01 (also timed out at 120s)** — all
   equal or worse.

## What I would try with more budget

The honest split-half harness (ES on one half of eval, score on the other, both directions) is what
unlocked the lossguide discovery, and it still showed a mild upward slope at the very end. With more
CPU I would: (a) push max_leaves/round-count along the lossguide frontier with a longer runtime limit
(lr≈0.01–0.0125 timed out at 120s — the ES curve wanted ~300+ rounds of 8192-leaf trees, so a
faster implementation via `xgb.train` + sparse int-encoded categories or `max_bin` tuning could fit
it in budget); (b) average 2–3 such champions (different seeds/max_bin) — runtime currently forbids
even a 2-model ensemble inside 120s, but with a 2× budget it typically adds +0.0005–0.001; (c) look
for transferable encodings of route/airport that do not use 2005 labels (e.g. schedule-structure
features: an airport's hourly departure counts in-year, carrier mix per hour), since everything
label-based failed; (d) tune `max_cat_to_onehot`/partition thresholds for Origin/Dest under lossguide,
which was never explored because each fit costs ~100s of the 4-thread budget.
