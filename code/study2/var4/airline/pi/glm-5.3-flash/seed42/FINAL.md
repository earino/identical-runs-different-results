# Final report — airline delay (XGBoost autoresearch)

**Best Eval AUC: 0.7251** (baseline 0.7141, +0.0110; hidden-holdout scorer uses the same `predict_proba(df)` path, validated `CONTRACT OK`).

Final model: average-in-log-odds of 16 XGBoost members (hist, depth ~6, lr 0.02–0.1 + 2 random-forest-mode
members) over a feature set built from the raw columns: DepTime → hour + cyclical sin/cos, native-categorical
Month/DayofMonth/DayOfWeek/carrier/Origin/Dest, and six *bounded-cardinality interaction categoricals* —
Carrier×Hour, Carrier×Month, Carrier×DistanceDecile, Hour×DistanceDecile, OriginTier×Hour, OriginTier×DistBin
(airport tiers = train-frequency quartiles). All statistics (category vocabularies, distance deciles, tier maps)
are fit on train only inside/next to `prepare()`, so `predict_proba` reproduces them on unseen data.

## Changes that mattered most
1. **Ensembling (single → 4 → 16 members)**: 0.7150 → 0.7251. Moderate diversity (subsample/colsample 0.7–0.95,
   lr 0.02–0.15, two RF-mode members) beat aggressive members (depth 5/7 or subsample 0.6 members hurt).
2. **Bounded carrier & hub interactions** (Carrier×Hour, Carrier×Month: +0.0024; Carrier×DistBin, Hour×DistBin,
   OriginTier×Hour: +0.002 total). Structural, dense levels (~200–1000 rows each) — they transfer across years.
3. **DepTime decomposition** (hour, cyclical sin/cos; +0.0009 with a 200-tree model vs 30-tree baseline).
4. **Capacity + full-data fit**: 200 trees @ lr 0.1, no early stopping (+0.0009 over 30-tree baseline);
   early stopping on an internal split *underfit* here.
5. **Distance deciles** for interactions (+0.0009 vs quintiles; 20 bins overfit).

## What did not help
1. **Target encodings** of carrier/origin/dest/route (0.7091): 2005 marginal delay rates do not transfer to 2006
   — the single biggest negative result; anything label-derived is non-stationary here.
2. **High-cardinality route categorical** (Origin→Dest, 4198 levels; 0.7015) and frequency/count encodings.
3. **Calendar×hour interactions** (DoW×Hour, Month×Hour, Carrier×DoW) and heavy regularization
   (subsample/colsample/mcw bundles: −0.01) or depth ≠ 6.

## With more budget
Stack a level-2 XGBoost on out-of-fold member predictions (proper stacking instead of equal-weight logit
blending); search member mix (RF share, per-member feature views); test origin×dest routes restricted to
frequent pairs with rare-route grouping; tune the interaction set per member (feature-view diversity); and
re-check target encodings computed *within 2006-style rolling windows* — but the year-shift evidence says
structural-only features are the safer bet.
