# FINAL — airline dep-delay AUC maximization

**Best Eval AUC: 0.7490** (baseline 0.7141, +0.0349 over 40 experiments).
Final model: `8ab8292` (HEAD of `experiment`). Contract validated (`CONTRACT OK`,
predict_proba reproduces 0.7490 on eval.csv with the target column removed).

## The 5 changes that mattered most

1. **Two-stage fit** (+0.004 alone, enabled everything else): one 90/10 run with early
   stopping finds the best iteration count; then refit on ALL 100k rows at that count.
   Single-fit early stopping was throwing away 10% of the data and stopping early.
2. **Deep trees + low learning rate**: max_depth 24→28 with lr 0.02, subsample/colsample
   0.7, max_bin 512, ~200 trees found by ES. The delay signal (hour-of-day above all)
   is a high-order interaction that shallow/tidy configs underfit badly
   (0.7141 → ~0.74 through the depth ladder alone).
3. **Dropping Month** (+0.0016): month-of-year rates flip between 2005 and 2006;
   keeping Month let the model memorize 2005 seasonality. Classic time-shifted-data fix.
4. **2-seed ensemble** (+0.0007): average of two full-data refits differing only in seed.
   With subsample/colsample 0.7 the members are diverse enough to matter.
5. **Route/traffic features** (+0.0004): train-only medians for route/origin/dest distance
   and per-airport traffic counts, plus distance-vs-typical-route deltas.

## 3 things that did NOT help

1. **Target encoding Origin/Dest/route/carrier** (−0.005): 2005 airport-level rates
   transfer weakly to 2006 (rate correlation only ~0.38). Same for hour×carrier TE (−0.003).
2. **Cyclic / wrapped time features** (sin/cos of day, minutes-since-5am): flat or worse —
   plain Hour + Minute + min_of_day is what works; the "wrap" logic at 24:00-26:00 hhmm
   values adds noise, not signal.
3. **Regularization and capacity beyond the peak**: min_child_weight/gamma/lambda
   cost −0.010; depth 32, 3-seed ensembles (both lr), refits at 1.2x ES count, and
   DayOfWeek removal all slightly worse. The model wants raw depth, not smoothing.

## With more budget

The biggest untapped lever is ensembling across *hyperparameters*, not just seeds —
the depth ladder (20/24/28) each scored within ~0.001, so averaging 3-4 configs with
varied depth/subsample would likely add +0.002-0.004 if runtime allowed (needs the
120s cap raised or a leaner stage-1). Second, a proper time-aware validation scheme
(e.g. validate on late-2005 months only) would make keep/drop decisions less exposed
to the 2005→2006 shift that killed every target-encoding attempt; I suspect some
airport-level signal is real but was rejected because eval-year noise swamped it.
Third, DepTime minute-level patterns within the 24:00-26:59 hhmm range (flights
"after midnight" of the previous day) deserve a dedicated indicator rather than a
wrapped continuous feature.
