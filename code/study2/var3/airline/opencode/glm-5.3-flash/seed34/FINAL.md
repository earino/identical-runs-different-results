# Final report — airline delay XGBoost (autoresearch)

**Best Eval AUC: 0.7512** (baseline 0.7141, +0.037). Final commit: 82d7348, validated `CONTRACT OK`.

## What mattered most

1. **Interaction target encodings (OOF)** — smoothed delay rates per carrier/origin/dest × time-of-day,
   plus route and origin×day-of-week. Fit on train only; training matrix uses 5-fold out-of-fold encodings
   to avoid self-leakage (in-sample TE scored *worse* than no TE: 0.7108 vs 0.7173).
2. **Kernel-smoothed time profiles** — replaced/augmented block TEs with Gaussian circular kernels over
   5-minute blocks (sigma 6–8 blocks ≈ 30–40 min window) per origin/dest/carrier and per route
   (route with its own width + heavier smoothing, alpha 30). This was the single biggest lever
   (0.7416 → 0.7512 across resolution/width experiments).
3. **Count/congestion features** — log flight counts per origin, dest, route, carrier, and per
   origin × 15-min block (hub volume / congestion proxies): 0.7350 → 0.7393.
4. **Diversified XGBoost ensemble** — 7 members (depths 5–10, subsample 0.7–0.9, distinct seeds),
   800 trees @ lr 0.03, probabilities averaged: 0.7327 → 0.7350, and it compounds with every feature gain.
5. **Basic time features** — parsing c-<n> date strings to ints, hour/minute from DepTime,
   sin/cos of time-of-day, log-distance: 0.7141 → 0.7173.

## What did not help

- **In-sample (leaky) target encoding** (0.7108) — must use OOF for train rows.
- **Month/season interactions** (origin_month, month×block, day-of-year sin/cos: 0.7307–0.7392) and
  dow×block TE (0.7318) — no signal beyond what trees already extract; diluted the feature set.
- **Distance×hour TE + date-volume counts** (0.7332) and big capacity jumps (depth 10 single model,
  min_child_weight/gamma regularization, 1000 trees @ lr 0.02 — all equal or worse).

## With more budget

- Sweep the route-kernel width/alpha grid more finely and give entity kernels per-entity widths
  (busy hubs vs small airports likely want different smoothing).
- Multi-scale kernel blend (10/20/30-min windows summed) instead of single width.
- Larger ensembles with feature-subset diversity, and XGBoost-only stacking of member OOF predictions.
- Recency-weighted training samples (month-based) to adapt 2005-fit encodings to 2006+ drift.
