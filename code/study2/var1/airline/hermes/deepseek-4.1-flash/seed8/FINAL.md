# FINAL — autoresearch XGBoost (airline delay, 2005 train → 2006 eval)

**Best Eval AUC: 0.7487** — `train.py` at HEAD (`be3842e`), validated: `./validate.sh` prints `CONTRACT OK`
and reproduces 0.7487 through `predict_proba` on `eval.csv` with the target column removed.

Budget: 40/40 experiments, ~150 min wall clock, 8910/18000 CPU-seconds.
Baseline (unmodified `train.py`) was 0.7141, so the run adds **+0.0346 AUC**.

## What the final model is

The average of 16 XGBoost classifiers spanning tree depths 1–16 (`learning_rate=0.16`,
`n_estimators=200`, `min_child_weight=0`, `hist`, native categoricals), trained on 100k 2005 rows.
Features: `DepTime`, `UniqueCarrier`, `Origin`, `Dest`, `Distance` plus train-only log frequency
counts for `Origin`/`Dest`/`UniqueCarrier`. Runtime ~75s (fit 58s, eval predict 14s).

## Changes that mattered most

1. **Averaging over a wide capacity grid instead of one tuned model.** train (2005) and eval/holdout
   (2006) are year-shifted: per-carrier delay rates move a lot between the two years (AS 0.64 → 0.52,
   TZ 0.38 → 0.50), so a single depth/step-size choice is a year-transfer gamble. Depth-diverse
   averaging was the single biggest lever: 0.7141 (baseline, depth 6) → 0.7198 (depth × lr grid) →
   0.7238 (depths 2–7) → 0.7264 (depths 1–10) → 0.7407 (depths 1–16).
2. **Dropping the calendar columns** (`Month`, `DayofMonth`, `DayOfWeek`). Ablation: 0.7175 without
   them vs 0.7170 with them — they are pure noise on a year-shifted split. Six fewer features and no
   loss, so the final model is simpler for it.
3. **`min_child_weight=0`** (removing the leaf-weight floor). This was the second-largest lever and
   shows up at every depth: a depth-12 probe scored 0.7388 at `min_child_weight=1` vs 0.7322 at 10;
   applied to the whole ensemble, `mcw=1` gave 0.7467 and `mcw=0` gave 0.7483.
4. **Trimming rounds to 200** with `mcw=0`: 0.7487 at ~75s vs 0.7483 at ~111s (n=300) — equal or
   slightly better AUC with a comfortable margin under the 120s experiment cap.
5. **Keeping `Origin`/`Dest` as the primary carriers of signal plus log frequency counts.** Removing
   `Origin` costs 0.008 and `Origin`+`Dest` costs 0.018 — they dominate every other column.

## What did not help

1. **Target encoding** of `Origin`/`Dest`/carrier/route/month-day (smoothed, out-of-fold): 0.6950, a
   large loss. The per-category delay rates are exactly the part of the distribution that moves
   between 2005 and 2006, so encoding them into the features imports year-specific bias.
2. **Route as a categorical** (`Origin_Dest`, ~5k levels) and the calendar re-additions: 0.707 and
   0.708. High-cardinality target-derived structure overfits the training year; likewise the
   hub-structure features (carrier × airport counts/shares) were exactly neutral (0.7413 vs 0.7413).
3. **Sampling-based diversity**: row/column subsampling (`subsample`/`colsample_bytree` 0.7–0.9) hurt
   both as single models and as ensemble members, and DART / random-subspace members were worse than
   depthwise members (0.7396 / 0.7408 vs 0.7414). More members is not the same as better members.
4. **Extra boosting rounds and hyperparameter micro-tuning** (`max_bin=64`, `gamma=5`,
   `colsample=0.8`, 600 rounds, `max_cat_to_onehot`): all neutral or negative on the year-shifted eval.

## With more budget

Two directions look most promising. First, a proper depth-range extension with a time-aware stopping
rule: every capacity extension (depths 1–10 → 1–12 → 1–16) kept paying, but depths 17–20 at
`mcw=0`/300 rounds blew the 120s cap, so the grid was cut on runtime rather than on evidence — a
cheaper deep counterpart (fewer rounds or `lossguide` geometry at high `max_leaves`) would settle
whether the trend continues. Second, model selection should stop being a single-split gamble: with
only train (2005) and eval (2006) and no year-shifted internal validation, every keep/discard call
rode on ±0.0025 AUC of eval noise, and the last third of the run was picking between statistically
indistinguishable ensembles. A validation scheme built from within-2005 time blocks, used to weight
or prune ensemble members and choose the depth grid, would replace that noise-chasing with a real
criterion — the current artifact is defensible mainly because averaging many capacities is robust
regardless of which member is best.
