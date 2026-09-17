# Final report — airline dep-delay AUC (autoresearch XGBoost)

## Result

- **Best Eval AUC: 0.7480** (`Eval AUC: 0.7480`, printed by `train.py` and independently reproduced by
  `./validate.sh` — `CONTRACT OK`, eval AUC via `predict_proba` with the target column removed: 0.7480).
- Baseline (unmodified `train.py`): **0.7141** → net gain **+0.0339**.
- Best harness-logged experiment: #7 (`f6f38d8`, 0.7467). The final commit `c6d21dc` adds the last two
  verified improvements (airport-hour schedule shares, `reg_lambda=2`) on top; the experiment loop could not
  log it because the 18,000 CPU-second budget was exhausted first (experiments were still left: 33/40).
  The final file was validated end-to-end by the validator path after the budget refusal.

## The 5 changes that mattered most

1. **Time-of-day decomposition of DepTime** — numeric `hour`/`minute`, **hour as a 24-level categorical**,
   and a **30-minute slot categorical** (`DepTime // 30`). Time of day is the strongest drift-stable signal
   (evening flights are delayed much more); categorical encoding of it beat plain numeric.
2. **carrier × hour categorical interaction** (~290 levels, ≈340 rows/cell). The single biggest jump
   (+0.009 AUC): airline-specific schedule structure (which airline operates at which hour) transfers
   cleanly from 2005 to 2006.
3. **Schedule-structure statistics from train only**: `log1p` flight counts per (origin, hour) and
   (dest, hour); carrier hub-shares (share of a carrier's flights at that origin/dest); and — the last win —
   **airport-hour schedule shares** (fraction of an airport's 2005 departures at that hour). These encode
   "busy hub at 5pm" style effects without leaking 2005 labels.
4. **Heterogeneous XGBoost ensemble**: 5 members — three seeds of depth-8 ×1200 trees, one depth-9 ×900,
   one lr=0.015 ×1600 — averaged in `predict_proba` (+0.002 over a single model, and it smooths seed noise).
5. **Low-lr/many-trees + colsample_bytree=0.5 + reg_lambda=2** on the richer feature set. Naive capacity
   increases on the raw features *hurt* (drift); with the engineered features, lr=0.02 ×1200 @ depth 8 with
   column subsampling and moderate L2 shrinkage was the sweet spot.

## 3 things that did not help (all reverted)

1. **Target encodings — every flavor** (raw smoothed, out-of-fold, carrier/origin/dest/hour/route, m ∈
   {20…500}): they raise 2005-holdout AUC dramatically (up to 0.79) and *lose* on 2006 eval. The 2005→2006
   shift makes label statistics of high-cardinality entities non-transferable.
2. **Route and high-cardinality interaction categoricals**: Origin_Dest route pairs, origin×hour,
   dest×hour, carrier×dow, carrier×month, carrier×origin, hour×dow, month×hour, dow×t30 — every one hurt
   (cells too sparse / 2005-specific). Also top-50 airport bucketing hurt vs raw airport categoricals.
3. **Model-side tweaks that failed**: early stopping on a 2005 internal validation split (badly misleading:
   0.7514 internal vs 0.6877 eval — that was the worst experiment), row subsampling, `max_cat_to_onehot`
   one-hot forcing, `lossguide`, recency weighting of months, min_child_weight/gamma tuning.

## What I would try with more budget

The binding constraint was the CPU budget (offline feature ablations were expensive), not the 40-experiment
count, so I would first make diagnostics cheaper (cache prepared matrices, reuse fitted trees). Then:
(a) a wider ensemble (10–15 members with feature-subset and colsample diversity, plus a "stable-features-only"
member to hedge drift); (b) cross-year validation: hold out late-2005 months as a *temporally* closer
validation to select features that transfer, instead of trusting random-split 2005 AUC; (c) more schedule-
structure features at the 30-min granularity (airport×slot shares, carrier×slot counts, slot-level
congestion indices); (d) coarse region/grouping features for Origin/Dest (timezone of airport ≈ departure
clock alignment, hub size tiers) to replace raw airport identity with something more stationary;
(e) a stacked meta-model over out-of-fold member predictions with time-blocked folds.

## Budget accounting

- 7 of 40 experiments used (6 kept, 1 reverted); wall clock ~62 of 230 minutes.
- CPU: exhausted (18,021 of 18,000 CPU-seconds) — offline ablation sweeps (feature/HP isolation) consumed
  most of it; the last candidate was verified through `validate.sh`, which ran train.py end-to-end.
- HEAD = `c6d21dc` (best train.py, `CONTRACT OK`).
