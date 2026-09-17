# FINAL — airline delay (autoresearch XGBoost)

**Best Eval AUC: 0.7560** (experiment #18, commit `f1eadf1`), up from the 0.7141 baseline
(+0.0419). `./validate.sh` prints `CONTRACT OK` and reproduces 0.7560 through `predict_proba`.

## Final architecture

`train.py` builds a lean feature set and averages a 12-member XGBoost ensemble
(4 depth-8 members + 8 lossguide/96-leaf members, distinct seeds; every member is
`colsample_bynode`-regularized). All feature engineering lives in `prepare()`, which
`predict_proba()` applies to unseen rows; all category levels are fit on `data/train.csv` only.

## Changes that mattered most

1. **Time-of-day features from DepTime** — minutes-since-midnight plus 8 sin/cos harmonics of
   the daily cycle. The delay rate climbs from ~4% at 5am to ~80%+ late evening; the harmonics
   give the trees a smooth, year-stable encoding of that curve (0.7141 → ~0.735 with the ensemble).
2. **Carrier × hour-of-day interaction category** — the single biggest jump (+0.008): evening
   congestion is very carrier-specific (hub-and-spoke banks), and giving the model direct access
   to `carrier|hour` cells beat letting it rediscover the interaction under heavy column sampling.
3. **Lean feature set** — dropping Month and DayofMonth (and everything month/season-like) gained
   +0.006: with `colsample_bynode=0.2` every weak column dilutes the useful ones, and 2005's
   within-year patterns did not transfer to 2006 anyway (time-separated split).
4. **Distance re-expressions** — `log(Distance)` and `log(Distance) × hour` added ~+0.002 total.
5. **Low `colsample_bynode` (0.2-0.25) + seed-averaged ensemble with structural diversity**
   (deep members + lossguide members, rebalanced toward lossguide) — each step of regularization
   and member diversity added ~+0.001-0.002 (0.7349 → 0.7560).

## Things that did NOT help (all reverted or rejected)

1. **Target/empirical-Bayes encodings** of Origin/Dest/Route/Carrier/carrier×hour — hurt badly
   (0.70-0.71): delay-rate statistics fitted on 2005 do not survive the shift to 2006.
2. **High-cardinality raw interactions** — Route (Origin-Dest), hour-of-week, month×hour,
   origin×hour, route×daypart: all diluted the column sampling and/or overfit 2005.
3. **More capacity without regularization** — 200-400 deep trees, max_bin 512/1024, deeper d10-d12
   members, larger learning rates: consistently worse than the regularized shallow-ish members.
   Busyness/hub count features and holiday flags were also neutral-to-harmful.

## What I would try with more budget

The plateau at ~0.756 despite train-AUC < eval-AUC gap of only ~2 points suggests the feature
ceiling, not model capacity, binds. I would (a) search for more *shift-stable* interaction
categories of the carrier×hour kind — e.g., carrier×hour×dow, origin×daypart fitted as raw
categories with careful dilution control; (b) re-run the lossguide-member rebalance trend further
(2 deep + 10 lossguide, or a wider leaves/rounds sweep per member); (c) test a two-stage model
where a second XGBoost fits the residuals of the first with different feature emphasis; and
(d) use repeated time-ordered splits of the 2005 data to validate features without touching
2006, so that keep/discard decisions depend less on the single eval slice. My biggest structural
regret is that fitted-statistic features failed wholesale — a *hierarchical shrinkage* encoding
(e.g., Origin pulled toward its region/carrier mean with strong priors) might transfer better
than the flat encodings I tried.
