# FINAL — airline departure-delay AUC

**Best Eval AUC: 0.7398** (experiment #40, `train.py` at HEAD).
Baseline was 0.7141, so the loop added **+0.0257 AUC** over 40 experiments (CPU ~856/18000 s, wall ~9 min).

## Changes that mattered most

1. **Treat `Month`, `DayofMonth`, `DayOfWeek` as ordered numerics** (`c-<n>` → n), not unordered
   categoricals. These columns are already ordinal; XGBoost's partition splits on unordered categories
   let it memorize arbitrary month/day groupings that do not transfer from 2005 to 2006. (+0.003 at the time.)
2. **Collapse rare `Origin`/`Dest` levels** into a single `RARE` category (kept levels with ≥400 training
   rows). Airport-specific delay rates correlate only ~0.37 between 2005 and 2006, so fine-grained airport
   identities are a major source of temporal overfitting. This was the single most reliable lever (+0.004 total;
   threshold sweep 50→100→200→400 improved monotonically).
3. **Time-of-day derived features** from `DepTime`: `dep_hour`, `dep_minute`, `dep_time_min`, and daily
   `sin`/`cos`. The delay rate rises monotonically from ~4 % at 5 am to ~82 % at 11 pm and is extremely
   stable across years (per-hour rate correlation 0.96). These features only paid off *after* (1) and (2)
   cleaned up the noisy encodings, then contributed +0.003.
4. **Much larger tree depth.** With the overfitting sources removed, eval AUC rose monotonically with depth:
   5→6→7→8→10→14→20→unlimited gave 0.7247 → 0.7398. The model was underfitting the
   time-of-day × airport × calendar interactions, not overfitting them.

## Things that did NOT help

- **Dropping `Origin`/`Dest`**: eval fell to 0.7014 — even though their raw rates transfer poorly, they carry
  useful signal (geography/route effects interacting with departure time).
- **`Route` (Origin_Dest) categoricals** and **frequency encodings** of Origin/Dest/Carrier/Route:
  neutral-to-negative (+uncertainty), likely redundant with the categorical features.
- **Explicit pairwise interaction categoricals** (carrier×DOW, carrier×month, month×DOW) and **monotone
  constraints** on `DepTime`: both hurt. The constraint was too crude for a feature whose delay profile is
  roughly monotone but has a red-eye discontinuity.
- **DART booster** (0.7163, 20× slower) and **seed-averaged ensembles** (no gain over a single model).

## What I would try with more budget

The strongest signal in the closing experiments was that depth was still the binding constraint, so I would
push capacity further (deeper trees / more rounds with lower learning rate) now that rare-category overfitting
is controlled, and re-sweep regularization (`min_child_weight`, `reg_lambda`, `subsample`) at that depth
rather than at depth 5. After that I would explore out-of-fold target encoding of `Origin`/`Dest`/`Route`
smoothed toward the global rate, and a small ensemble over rare-collapse thresholds (e.g. 200/400/600) to
average away the identity-overfitting that the hidden 1 m-row 2006 holdout will punish. All of this should be
selected with a *time-based* internal split (late-2005 → early-2006 proxy) rather than eval.csv, since the
remaining differences are close to the ~0.0015 AUC standard error of a 100 k eval set.
