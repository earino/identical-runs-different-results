# autoresearch XGBoost — airline dep_delayed_15min — FINAL

**Best Eval AUC: 0.7330** (baseline 0.7141, +0.0189), by the 6-member XGBoost ensemble in
commit `8965195` (experiment #39: "minute + tod harmonic features"). Validated: `CONTRACT OK`.

## Changes that mattered most

1. **Fine time resolution + small leaves.** Raising `max_bin` 256 → 1024 and dropping
   `min_child_weight` 100 → 10-25 was the single biggest modeling win (0.7142 → ~0.725 single
   model). Departure-delay risk lives in fine minute-of-day structure.
2. **The `minute` feature** (DepTime % 100) + **cyclical harmonics**: sin/cos of minute at 60- and
   30-minute periods, plus 24h and 12h harmonics of time-of-day (0.725 → 0.728 single, 0.7330
   ensemble). Bank-departure structure within the hour was invisible to trees without it.
3. **Diverse XGBoost ensemble** (5-6 members, mean of probabilities): depth-wise d12/d16 plus
   lossguide (max_leaves 256-512), different seeds; +0.001-0.002 over any single model.
4. **Low colsample (0.5) + colsample_bynode 0.6-0.7, subsample 0.75, lr 0.03, ES on AUC.** Low
   column sampling with deep trees was consistently the productive region; the best single member
   was lossguide/512 leaves/mcw10/bynode 0.7 (0.7296 single).
5. **Early stopping on AUC** (not logloss) on a 30k eval subsample — logloss stopping cuts trees
   well before the AUC peak; the subsample keeps 6 members within the 120s cap.

## Things that did not help

1. **Route as a raw 5k-cardinality categorical** — optimal partitioning overfit (~20 rows per
   route) and dragged early stopping down (0.7145 → 0.7060).
2. **Smoothed out-of-fold target encoding** (route/origin/dest/carrier) and **congestion
   statistics** (flights per origin/hour, hour share): 2005-fitted delay-rate statistics did not
   transfer to 2006 — slight negative.
3. **DART** (2-3x slower, one model ate the whole 120s budget), **lr 0.02 / max_bin 2048** (no
   gain over lr .03 / 1024), **AUC-weighted and rank-blended ensemble averaging** (equal to plain
   mean at higher cost), **interaction categoricals** (carrier×hour etc. — deep trees find them).

## With more budget

The signal is dominated by scheduled-time-of-day; the ceiling with these 8 columns is feature-
limited, not capacity-limited. I would (a) grid the harmonic family more finely (per-airport
phase offsets, minute-bank categoricals), (b) build a larger 10-15 member ensemble now that ES
subsample tricks are known, with per-feature max_bin, (c) revisit route/airport statistics in a
year-shift-robust form: differences from the *network-wide* hourly delay rate rather than absolute
rates, so carrier/airport effects are expressed as offsets that are more stable 2005→2006, and
(d) try `one_drop`/dropout-tree boosting with a fixed round count sized for the 120s cap.
