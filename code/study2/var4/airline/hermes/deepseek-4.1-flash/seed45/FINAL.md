# Final report — airline departure-delay classifier (XGBoost)

**Best Eval AUC: 0.7407** (commit `298d903`, experiment #39 of 40). Baseline: 0.7141.
All 40 experiments, 192 minutes of the 230-minute wall clock, and ~5.9k of the 18k CPU-second
budget were used; `./validate.sh` prints `CONTRACT OK` and reproduces 0.7407 through
`predict_proba(df)` on eval rows with the target column removed.

Final model: an ensemble of 36 `XGBClassifier` members (12 hyperparameter configurations x 3 seeds,
each trained on a different random 80% of the rows), all fitted on smoothed target/frequency
encodings of the raw `airline` columns.

## Changes that mattered most

1. **Compound smoothed target encodings** (+0.0119 over baseline: 0.7141 -> 0.7260 by exp #12).
   Two-way keys (carrier x hour, origin x hour, dest x hour, route x hour, carrier x route) with
   additive smoothing toward the training prior were by far the largest single lever. Encodings are
   fit on training keys/labels only; training rows get out-of-fold values (10-fold) so the members
   never see their own labels, while `predict_proba` applies the full-train maps.
2. **Dropping raw high-cardinality ids** (+0.0022 for Origin/Dest, exp #30/#31). Feeding the raw
   airport/carrier categoricals to the trees overfits the 2005 slice; the smoothed encodings carry
   the same signal in a form that transfers to 2006. This also made every later model faster.
3. **Diverse member ensembles** (+0.0065 across exp #8 -> #39: 0.7141-baseline ladder to 0.7407).
   Depth 3-12, colsample 0.3-0.9, lr 0.05-0.15 and round counts varied per member; per-member row
   subsampling (0.8) added diversity beyond seed changes alone. Very deep members (depth 10-12)
   started helping only after the raw ids were removed (+0.0005).
4. **Time-of-day features** (+0.0034 for the first version, exp #2). Parsing `DepTime` hhmm into
   minutes-since-midnight plus circular hour features; hour is the strongest raw driver.
5. **Heavier encoding smoothing** (k x3, +0.0003, exp #34) and **schedule-structure statistics**
   (hub share ratios, distinct-carrier-per-route / routes-per-airport counts, +0.0003, exp #33).
   Both are year-stable quantities, which is why they transfer.

## Things that did not help (reverted)

1. **Capacity/early stopping on in-sample validation** (exp #3, #6, #7). A depth-7 lr-0.05 model with
   early stopping kept improving its 2005 validation AUC for 1698 rounds while 2006 eval AUC fell to
   0.7121. The 2005/2006 shift punishes capacity that is tuned on a same-year holdout; small
   explicitly-limited trees blended together generalize better.
2. **Calendar-position features** (exp #16, #21). Month as a categorical, month target encoding and
   month cyclical terms all hurt or tied (0.7307, 0.7348): seasonal delay propensity is not stable
   across years. Dropping them and keeping dom/dow/weekend was the right trade.
3. **Sparse 3-way keys** (exp #20, 0.7350) and **DART members** (exp #28, timed out; DART needs far
   more than 120 s for a useful ensemble here). Sparse keys just re-encode existing signal with more
   noise, and DART is not affordable under the per-experiment time cap.

## What I would try with more budget

The single most informative result is that every time I removed an overfitting channel — raw ids,
calendar terms, in-sample capacity tuning — eval AUC went up, so the remaining headroom is probably
in *sharper but more robust* estimates rather than more features. Concretely: (1) replace the hand-tuned
additive smoothing with a properly nested encoding, e.g. per-key credibility weights or a small
hierarchical/Bayesian shrinkage fit by cross-validation, since k x3 beat k and k x9, implying an optimum
exists that I only bracketed; (2) fit ensemble member weights on out-of-fold predictions rather than
averaging uniformly (a level-2 logistic blend on OOF predictions instead of logistic regression on
eval, which would leak); (3) search the deep-member region more finely — the depth 10-12 members were
just starting to pay off when the budget ran out (the depth 10-14 attempt timed out); (4) train members
on feature subsets (e.g. encodings-only vs. hourly-only views) for stronger decorrelation; and
(5) revisit the `DepTime` outliers (values like 2620 that are not valid hhmm) with a smarter recovery
rule instead of mapping them to missing.
