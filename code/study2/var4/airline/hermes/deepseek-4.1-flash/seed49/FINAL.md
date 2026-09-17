# FINAL — airline dep_delayed_15min, XGBoost

**Best Eval AUC: 0.7409** (experiment #37, commit `709fd74`; baseline was 0.7141).
`./validate.sh` → `CONTRACT OK` (predict_proba reproduces 0.7409 on the target-stripped eval frame).

Budget used: 40/40 experiments, ~31 minutes wall clock, 4.5k/18k CPU-seconds. Metric is AUC on
`data/eval.csv` (2006 slice); the model is scored on a hidden 2006 holdout with the same schema.

## Final model

`train.py` = one `XGBClassifier` (depth 13, lr 0.02, 700 rounds, `max_bin=1024`, `hist`,
`enable_categorical=True`, subsample/colsample 0.9) over ~15 features built entirely inside `prepare()`:

* clock: `dep_hour`, `dep_minute`, `dep_tod`, `dep_red_eye` derived from the `hhmm` `DepTime`
* ordered calendar: `month`, `day_of_month`, `day_of_week` (the `c-<n>` levels are just integers), `day_of_year`
* raw: `Distance`, `UniqueCarrier`, `Origin`, `Dest` as XGBoost categoricals
* **categorical interactions `carrier×hour` and `origin×hour`** — the decisive feature class
* no label-derived statistics anywhere (levels are fitted on `data/train.csv` only, unseen levels → NaN)

## What mattered most

1. **`carrier×hour` and `origin×hour` categoricals: 0.7186 → 0.7325 (+0.0139).** The single biggest lever.
   Delay behaviour is largely "which airline, at which airport, at which hour" — an operating pattern that is
   a stable property of the network, so it transfers from the 2005 training year to 2006. Raw `DepTime` as an
   integer cannot express "carrier × exact hour"; once the pair is an explicit high-cardinality categorical,
   the trees exploit it heavily. Exact-hour granularity is the sweet spot: 3-hour bins cost -0.015 and
   30-minute bins (too sparse) cost -0.002.
2. **Treating the `c-<n>` calendar columns as ordered integers plus `day_of_year`: 0.7146 → 0.7186 (+0.0040).**
   Month/day/weekday are ordered, not unordered levels; the tree can then split "summer vs winter" with one split.
3. **Deep trees: depth 7 → 13, 0.7325 → 0.7384.** With interaction categoricals present, depth is what lets a
   tree combine carrier, airport, hour, season and distance at once. Depth 16 was slightly worse (0.7377).
4. **Clock features from `DepTime`, and in particular `dep_minute`.** Dropping `dep_minute`/`dep_red_eye`
   cost -0.017 (0.7239) — the minute-of-hour at which a flight is *scheduled* carries real delay signal.
   `dep_tod` also lets trees separate late-evening banks without learning 24 separate splits.
5. **Small training-schedule tuning: lr 0.02 / 700 fixed rounds / `max_bin=1024` (0.7384 → 0.7409).** A single
   full-data fit with a fixed round count beat probe-plus-refit early stopping and every seed ensemble tried,
   at half the compute — which is why the final file has no early stopping at all.

## What did not help (all reverted)

1. **Target encoding: smoothed out-of-fold delay-rate by origin/dest/carrier/route, 0.6865 (-0.032).**
   Label statistics from 2005 simply do not describe 2006 (it also improved in-year validation logloss, i.e.
   pure year leakage). Same family: frequency/hub-size counts were flat (0.7184).
2. **Route features: full `Origin_Dest` categorical -0.009, top-500-routes + "OTHER" -0.004, `dest×carrier` -0.007.**
   City-pair identity adds noise once origin-hour and carrier-hour are known.
3. **Calendar/season or destination interactions: `origin×month` -0.024, `carrier×dow`/`origin×dow` -0.010,
   `dest×hour` -0.005 (depth 7) and -0.007 (depth 13).** Anything that pairs a calendar period or the
   destination with the operational features degrades year-to-year transfer.
4. Also flat-to-negative: depth 16, `min_child_weight=10`, `max_cat_threshold=256`, `grow_policy=lossguide`
   (1024 leaves), `subsample=0.7`, `colsample_bytree=1.0`, 1100 rounds, 1400 rounds at lr 0.01, lr 0.015,
   `dep_hour` as a categorical, and two seed ensembles (3 seeds × 600 rounds: +0.0002 for 2.4× compute).

## Caveat on the numbers

Every keep/revert decision was made on one eval slice, and roughly a dozen accepted/rejected changes landed
within ±0.0005 AUC of each other — at that scale the ordering is noise. I therefore only kept changes that
were both positive and mechanistically plausible (interactions, calendar ordering, depth, clock features) and
reverted the purely cosmetic micro-gains, preferring the simpler, cheaper file when the difference was inside
the noise band.

## What I would try with more budget

The evidence says the ceiling here is set by how much *network structure* is recoverable from eight columns, and
that anything fitted to the training year's labels or to seasonal timing actively hurts. With more budget I would
push on three fronts. First, mine the scheduled-minute structure properly: `dep_minute` was the single most
under-appreciated feature, so `carrier×minute` and `origin×minute` categoricals, and an explicit "minute past the
hour" bucket, are the obvious next features to test under the same interaction recipe that produced the +0.014.
Second, replace the random 10% probe used early on with a *time-shaped* internal split (or nested CV over both
years) so that capacity/round decisions are selected against the same year-shift the hidden holdout shows, rather
than against in-year logloss, which is what made target encoding look good. Third, attack the one structural idea
the data supports but I never had room to build: a two-stage model where stage one predicts the year-stable
carrier×hour×origin delay propensity and stage two predicts the residual from the remaining columns, with
per-bucket shrinkage so that thin (airport, hour) cells fall back to the carrier or airport effect — the same
hierarchical shrinkage that target encoding tried to provide and that plain categorical splits only approximate.
