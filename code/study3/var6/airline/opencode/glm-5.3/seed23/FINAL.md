# Final report

**Best Eval AUC: 0.7440** (experiment #18, commit `8022416`; validated `CONTRACT OK`).

Baseline (30 depth-6 trees, raw features) was **0.7141** — net improvement **+0.0299**.

## Changes that mattered most

1. **Multi-view bagged ensembling** — average many small XGBoost models trained on *decorrelated feature
   representations* of the same row: raw categoricals (`base`), origin/dest smooth target encodings (`ot`),
   origin×hour / dest×hour encodings without airport cats (`no_od`), and an all-numeric target-encoding soup
   (`fullte`). View diversity, not model count, drove the early gains (0.7213 → 0.7262).
2. **Out-of-fold target encodings** — TE features for training rows are computed from the *other* 4 folds
   (StratifiedKFold), while `predict_proba` uses the full-training map. Removing self-label leakage made TE
   features transfer across the 2005→2006 shift (origin×hour view solo: 0.7161 → 0.7235).
3. **`grow_policy="lossguide"` trees with many leaves** — the single biggest lever. Leaf-limited asymmetric
   trees (up to 512 leaves, `max_depth=16`, heavy row/col subsampling) transfer far better to the holdout year
   than depth-limited trees: 0.7272 → 0.7386 overnight, and re-tuning leaves under finer binning gave 0.7440.
   The view ranking reshuffled completely under this regime (raw-categorical `base` became the strongest view).
4. **Dropping time-unstable features from the main views** — Month/DayofMonth delay rates barely correlate
   across years (0.59/0.33); removing them from the primary views helped early on (0.7141 → 0.7213) while
   keeping stable DayOfWeek/hour/airport/carrier signal.
5. **`max_bin=512` + big leaves interaction** — finer histogram bins moved the optimal `max_leaves` upward;
   together +0.0026 (0.7414 → 0.7440). Within-view bagging (4 seeds, subsample 0.9, colsample 0.8).

## Things that did not help

1. **Stacked generalization** — a meta-learner on 2005 out-of-fold view predictions *underperformed equal
   averaging*: view reliabilities measured on 2005-OOF are nearly inverted under the 2006 shift
   (md view: OOF 0.7458 vs eval 0.7171; oh view: 0.7134 vs 0.7145).
2. **Alternative boosters/objectives** — `rank:pairwise` (0.6496 solo), DART members, and gblinear members all
   reduced mixture AUC; `max_bin=1024` matched 512 at +14s; leaves 256 was worse.
3. **More feature keys than the four committed views** — carrier×airport and airport×day-of-week TEs,
   temporal Month/DayofMonth TE views (modom diluted 0.7409 → 0.7403), recency weighting, bootstrap bags,
   TE fold-seed diversity, and per-view mcw/lambda polish were all neutral or slightly negative.

## With more budget

I would push the two live trends first: the leaf trend (320→384→448→512 gained at every step and had not
plateaued at the CPU cutoff) jointly with `max_bin` and tree count, using a cross-year validation scheme
(train on 2005 months 1–9, validate on Nov–Dec 2005) to tune for transfer rather than trusting the single
2006 eval slice; then re-run full view-family selection under the tuned lossguide regime (the ranking
reshuffled completely once tree shape changed, so more decorrelated representations — e.g. conditional
origin×hour×distance-bin TEs — may now earn their place), and finally scale up seeds per view for holdout
variance reduction, which the 120s training cap currently prevents.
