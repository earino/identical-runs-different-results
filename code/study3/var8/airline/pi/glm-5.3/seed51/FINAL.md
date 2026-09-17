# Final Report — autoresearch XGBoost (airline delay)

**Best Eval AUC: 0.7566** (experiment #15, commit `3aeb53b`, `./validate.sh` → CONTRACT OK, hidden-holdout
prediction path verified end-to-end). Baseline was **0.7141** — a total gain of **+0.0425 AUC**.

## Final model

16-member XGBoost ensemble (8 rare-bucketing thresholds × 2 depths), mean of predicted probabilities:
features = DepTime, Distance, UniqueCarrier, Origin (bucketed), Dest (bucketed), DayOfWeek,
carrier-share-at-origin; members = `XGBClassifier(n_estimators=110, max_depth∈{14,18}, learning_rate=0.05,
colsample_bytree=0.85, max_bin=384, max_cat_to_onehot=32, tree_method="hist", enable_categorical=True)`;
bucketing threshold ladder r ∈ {150, 200, 300, 400, 500, 700, 1000, 1500} (airports with < r train flights →
"OTHER"). Trains in ~55 s, whole experiment ~85 s (cap 120 s).

## The 5 changes that mattered most

1. **Drop Month / DayofMonth** (0.7141 → 0.7187): 2005 seasonal patterns do not transfer to 2006 — the single
   most important robustness decision, found by ablation.
2. **Rare-airport bucketing with a threshold ladder** (→ 0.7387): airports with few train flights are noise;
   mapping them to "OTHER" (train-count threshold) makes deep trees safe, and averaging over a ladder of
   thresholds gives the ensemble its diversity axis.
3. **The 16-member ensemble itself** (→ 0.7509): deep trees (d14–d18) + per-tree column subsampling
   (colsample_bytree=0.85) only work as an ensemble; the mean over the rare-ladder × depth grid was worth
   +0.012 over the best single member.
4. **max_cat_to_onehot=32** (→ 0.7566, +0.0043): one-hot-style splits for low-cardinality categoricals
   (DayOfWeek, UniqueCarrier) instead of partition splits — the last big find, and it made training *faster*
   too.
5. **Structural micro-gains kept along the way**: max_bin=384 (+0.0008) and the carrier-share-at-origin
   feature (+0.0002 at ensemble level but +0.0018 per member — count-based, no target leakage, so it should
   transfer to the hidden set).

## 3 things that did not help (all reverted or abandoned)

1. **Any 2005-fitted delay statistic** — target encodings of Origin/Dest/carrier (raw, smoothed, m-weighted,
   origin×hour, etc.), capacity models, temporal early stopping: all memorize 2005 and transfer badly to 2006.
2. **Row-bagging / subsample, jitter augmentation, cheap-member scaling** (n=50 lr=0.1, 40 members), snapshot
   averaging, per-member seeds — none beat the fixed 16-member recipe; subsampling in particular always hurt.
3. **DART boosting** — single members gained (+0.0022) but at 121 s/member it cannot fit the 120 s cap, and
   shrinking it (n80/d12) loses the entire gain. Also: route features, share-at-dest, entropy features,
   asymmetric origin/dest thresholds, one-sided bucketing, DepTime wrapping/binning — all flat or worse.

## What I would try with more budget

The 2005→2006 distribution shift is the binding constraint, so I would keep hunting for *structurally stable*
(transferable) signals rather than more capacity: e.g., learning per-airport delay profiles from *schedule*
geometry (departure-bank concentration, hub vs spoke typology) rather than target statistics; a
carrier-hub graph feature family (dominance shares at both endpoints, ratio forms); and smarter categorical
split control (the `max_cat_to_onehot` axis was only probed at 8/32/64 — the sweet spot around 16–48 deserves a
sweep at composition level). I would also re-examine the aggregation: per-member stacking weights fit by
out-of-time (temporal) cross-validation inside 2005, which avoids the target-leakage trap that killed plain
TE. Finally, DART at a size that fits the cap (e.g., n≈100, d14, rate_drop≈0.08, two members replacing two
gbtree members) is worth one careful experiment.

## Experiment log (official runs)

| # | change | Eval AUC | kept? |
|---|--------|----------|-------|
| 1 | baseline (30 trees, d6, native categoricals) | 0.7141 | ✔ |
| 2 | numeric time features + route categorical | 0.7089 | ✘ |
| 3 | + capacity model, early stopping | 0.7074 | ✘ |
| 4 | ablation: baseline features + capacity | 0.7103 | ✘ |
| 5 | drop Month/DayofMonth, d5 | 0.7187 | ✔ |
| 6 | rare-airport bucketing, d12/100tr/lr.05 | 0.7387 | ✔ |
| 7 | 16-member ensemble (rare ladder × depths) | 0.7509 | ✔ |
| 8 | max_bin=384 | 0.7517 | ✔ |
| 9 | carrier-share-at-origin feature | 0.7519 | ✔ |
| 10 | depth pair {14,18} | 0.7520 | ✔ |
| 11 | n_estimators=110 | 0.7523 | ✔ |
| 12 | mid-dense rare ladder | 0.7521 | ✘ |
| 13 | low-shifted rare ladder | 0.7518 | ✘ |
| 14 | weighted mean aggregation | 0.7523 | ✘ (tie) |
| 15 | **max_cat_to_onehot=32** | **0.7566** | ✔ (final) |

Budget exhausted via the CPU limit (18,000 Python CPU-seconds): 15 of 40 experiments used, ~128 min wall.
CPU was the binding constraint; every kept commit improved eval AUC and `HEAD` is the best experiment.
