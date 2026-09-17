# Final report — airline delay XGBoost

**Best Eval AUC: 0.7357** (commit 80c942d, reproduced twice: experiments #38 and #40).
Baseline: 0.7141 → **+0.0216 AUC**.

## Final architecture

Ensemble of 11 XGBoost models (9 hist depth-5/6/8 + 2 lossguide max_leaves 32/128; one per-seed
bagging-free, lr 0.05, subsample/colsample 0.7, min_child_weight 10), each early-stopped on a random
25% validation split (logloss) and refit on the full training data with trees capped at 200. Features
(inside `prepare()`, all statistics fit on train only): departure time-of-day features, cyclical
month/dow encodings, distance + log + decile bin, 10 smoothed out-of-fold target encodings
(Origin, Dest, carrier, 15-min time slot, Origin|slot, Dest|slot, dow|slot, exact DepTime, route, dow|hour),
log train-count features (incl. Origin|slot and Dest|slot traffic), and raw categorical columns.
Training rows use OOF-encoded TE values; unseen rows use full-train maps (unseen keys → global mean).

## Changes that mattered most

1. **Fine-granularity time target encodings (+0.0082 alone, exp #21→#22/#23):** switching the TE time
   key from hour-of-day to 15-minute DepTime buckets was the single biggest gain — DepTime carries a
   very strong, very fine-grained delay signal (delay rate 4%→83% across the day).
2. **Ensembling with feature-view diversity (+0.0022, exp #27):** adding 2 models trained without the
   TE columns to the 7-seed ensemble beat any seed/size tuning. Later swapping 2 hist members for
   lossguide trees added +0.0001 (exp #38).
3. **Smoothed OOF target encoding + count features (+0.002 before granularity tuning, exp #8→#10):**
   Origin/Dest/carrier/hour/route TE with m=25–100 smoothing, out-of-fold values for training rows;
   origin×slot / dest×slot traffic counts (+0.0012, exp #15).
4. **Early stopping + refit, with a hard tree cap (+0.0022, exp #5→#6→#10):** internal-val-optimal
   capacity systematically overfits the 2005→2006 year shift; capping trees at 150–250 (ES picks
   300–1000) recovers ~+0.002.
5. **Removing the raw route category (+0.0011, exp #3→#4):** a 3000-level categorical invites
   memorization; route info enters only via smoothed TE.

## Things that did not help

- Raw high-cardinality route category (exp #2–#3): worse than baseline.
- Higher capacity without caps — 600×depth-8 (exp #2), ES-on-AUC picking 552 trees (exp #6), lr 0.1 (exp #12).
- Carrier×hour / carrier×month TE (exp #17), origin/dest×dow TE (exp #30), exact-m smoothing m=10 (exp #24),
  min_child_weight 20 (exp #20), gamma 1 (exp #26), colsample_bynode 0.7 (exp #39), max_bin 512 (exp #33):
  all neutral-to-negative.
- Bagging members on 80% subsamples (exp #36) — slightly worse than full-data members; rank-averaging
  and 50/50 view weighting — identical to plain probability averaging.

## With more budget

The clearest remaining direction is **finer, better-smoothed time TE**: 5-minute slots with per-key
tuned smoothing (m swept per cardinality), TE on Origin|Dest|slot at coarser granularity, and
interaction TEs (distance-bin × slot). Second: a larger but time-budgeted ensemble (13+ members
timed out at 120s; a two-pass "ES once, refit all members" scheme would fit ~16 models). Third:
per-member hyperparameter diversity (learning rate, subsample) instead of seed-only, and a final
capacity re-sweep (tree cap 100–300 in steps of 50) with the final feature set, since the cap was
tuned on earlier, weaker features.
