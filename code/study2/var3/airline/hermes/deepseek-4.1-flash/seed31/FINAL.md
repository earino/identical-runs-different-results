# FINAL — airline delay prediction (XGBoost)

**Best Eval AUC: 0.7309** (held-out `data/eval.csv`, 2006 slice), from experiment #40 (`fa8b8bb`,
final file `baed2fb`, identical model). Baseline was 0.7141, so +0.0168 over 40 experiments.

The submitted `train.py` trains a 16-member XGBoost ensemble. `predict_proba(df)` reproduces the whole
pipeline on raw rows: every engineered feature is computed inside `prepare()` from statistics that were
fitted on `data/train.csv` only, so the hidden holdout gets exactly the same transformation.
`./validate.sh` prints `CONTRACT OK`, eval AUC 0.7309, in 99 s.

## Changes that mattered most

1. **Departure time-of-day decomposition** (exp 5, 0.7141 → 0.7177). `DepTime` (hhmm integer) is replaced by
   `dep_min` = minutes since midnight, plus hour, minute and sin/cos. Delay rate rises monotonically from
   0.04 at 05:00 to 0.82 at 23:00, so this single feature carries most of the available signal, and the
   raw hhmm encoding makes it needlessly hard to split.
2. **Smoothed delay-rate encodings of Carrier/Origin/Dest × hour** (exp 10, 0.7177 → 0.7195). Per-cell
   delay rates from 2005, smoothed toward the global prior (α = 300/500), added as three numeric columns
   while the raw categoricals stay. The hour interaction is stable year-over-year; the bare airport or
   route delay rate is not (see below).
3. **Lower learning rate with a fixed round count** (exp 15/34/39, 0.7195 → 0.7309 as part of the ensemble).
   `learning_rate=0.03` with 1300 rounds fits the dense balanced data hard with mild shrinkage, which beats
   both heavy regularization and very low learning rates.
4. **Ensembling over tree depth** (exp 20–27, 0.7199 → 0.7226). Averaging probabilities of models with
   `max_depth` from 2 to 8 is the only change that kept paying after single-model tuning plateaued.
5. **Ensembling over feature views** (exp 28–29, 0.7226 → 0.7265). Each member is trained on one of three
   views: all features, no delay-rate encodings, or no raw Origin/Dest/UniqueCarrier categoricals.
   This was the single largest jump after the time features, and the final model is 16 members spanning
   depths 2–12 across the three views.

## Things that did not help

1. **High-cardinality route features.** `Origin_Dest` as a categorical dropped eval AUC to 0.7065 (exp 6),
   and replacing the raw Origin/Dest/Carrier categoricals with smoothed target encodings dropped it to
   0.7054 (exp 9). Airport-level delay propensity is only ~0.25–0.38 correlated between half-years, so
   these features mostly memorize 2005.
2. **Regularization and capacity tweaks on the single model.** `subsample`/`colsample`/`min_child_weight`/
   `reg_lambda` (exp 8, 0.7156), seed-bagged bagging with subsample 0.8 (exp 13, 0.7154), `max_depth=8`
   (exp 14, 0.7175), `max_cat_to_onehot=32` (exp 18), `max_cat_threshold=128` (exp 19) all lost to the
   plain depth-6 configuration. Frequency encodings (exp 7) and DayOfWeek×hour / Month×hour rates (exp 17)
   were exactly neutral.
3. **Shortening the ensemble members to buy more of them.** 22 members at 700 rounds scored 0.7292 and 19
   members at 1100 rounds scored 0.7304, both below 16 members at 1300 rounds (0.7309) — member quality
   mattered more than member count once the views were diverse.

## What I would try with more budget

The evaluation metric is a single 100k-row AUC, whose standard error is roughly 0.0016, so anything below
~0.002 is close to noise and several of my keep/discard calls were coin flips. With more budget I would
first replace the noisy eval-driven decisions with a stabilised selection signal — e.g. repeatedly fit on
disjoint halves of 2005 and score on the other half, or hold out a fixed pseudo-2006 slice by month — then
use it to tune things I could only guess at here: the α smoothing of the delay-rate encodings, whether
`grow_policy="lossguide"` or a few DART members add real diversity, and the exact depth ladder. Second, I
would test proper out-of-fold target encoding of the hour cells (the current encodings are in-sample for
training rows, which flatters the internal fit), and a stacker over the member probabilities instead of a
plain average. Third, the most promising untried direction is more genuinely different feature views —
each added view paid more than any hyperparameter or extra member did — and a transductive-style use of
the unlabelled 2006 features is worth one careful look, provided it stays inside the contract.
