# FINAL — airline delay (XGBoost), autonomous run

**Best Eval AUC: 0.7644** (experiment #23, commit `e33922f`) on `data/eval.csv` (2006 slice).
Baseline for comparison: **0.7141** (experiment #1). Net gain **+0.0503 AUC**.
`./validate.sh` → `[validate] CONTRACT OK`; validation re-runs training and reproduces 0.7644 through
`predict_proba(df)` on `data/eval.csv` **with the target column removed**, so the feature path is
reproducible on unseen rows.

## Budget outcome

- 23 of 40 experiments used; ~143 minutes of wall clock left.
- The run ended on the **CPU budget** (≈17.7k of 18,000 Python CPU-seconds) with ~0.7k reserved for the
  mandatory final `validate.sh`. Stopping here was deliberate: spending the last CPU on another experiment
  would have made the required end-of-run validation fail, which scores the whole run as a failure.

## The 5 changes that mattered most

1. **Deep trees with strong column subsampling (+~0.025 at the time, experiments #7–#8).** Depth was the
   single dominant knob and it went the *opposite* way from the usual intuition: the depth-6 baseline was
   badly underfit, and a depth-24 model with `colsample_bytree≈0.5` beat every shallow/regularized
   alternative. Column subsampling (not row subsampling) is what keeps the deep trees from memorising
   2005-specific noise. Depths 8–40 with colsample 0.4–0.5 all land in the same good region.
2. **Dropping all calendar features (+0.008, experiment #9).** `Month`, `DayofMonth` and `DayOfWeek` as
   categoricals are *actively harmful* here: their seasonal/day-of-month splits are year-specific and do not
   transfer from 2005 to 2006. Every attempt to re-add calendar signal (month as a category, as a number, or
   as a sine/cosine pair; day-of-week cyclical; weekend flag) made things worse.
3. **Frequency counts and hub-size statistics (+0.005, experiments #9–#10).** Counts of carrier / origin /
   destination / city-pair in the training data, plus "number of distinct destinations per origin",
   "number of carriers per origin", etc. These are stable structural proxies for how big and how busy an
   airport or carrier is, and they transfer across years far better than the identity categories themselves.
4. **Estimated arrival-time features (+0.004, experiment #11).** Scheduled arrival approximated as
   departure clock time + distance/460 mph + 0.5 h. Late-evening arrivals accumulate the day's delay, and
   this was the best single feature addition of the run.
5. **Bag width and diversity (+~0.005, experiments #13–#17, #22).** Averaging many deep models over a grid
   of depths (8…40, two column-subsample regimes) and seeds; half the bag additionally drops the
   high-cardinality `Origin`/`Dest` identity columns so the two sub-bags make different mistakes.
   `max_bin=128` (experiment #22) made fitting cheap enough to raise the tree count per model and was worth
   another +0.0007.

## The 3 things that did not help

1. **Target encoding of Origin / Dest / carrier / route, and route as a categorical** (0.7058 and 0.6999 vs
   0.7184 at the time). Airport and route delay propensities drift year to year; anything that memorises
   *which* airport/route it is, rather than how big it is, actively hurts. This was the clearest signal in
   the whole run about what kind of task this is.
2. **Extra capacity without column subsampling.** Early stopping on a 2005 holdout picked ~371 trees at
   depth 7, which scored 0.7020 — worse than the 30-tree baseline. Internal (same-year) validation
   systematically over-estimated capacity; only eval-on-a-later-year exposed it.
3. **A pile of plausible micro-changes**: row `subsample<1` (hurts alone and inside the bag), `dart`
   booster, averaging in rank space instead of probability space, adding raw `DepTime` alongside the clock
   features, minute-roundness flags, carrier×month and carrier×hour interactions, distance bins, and
   `colsample_bynode`. All within ±0.001 of their reference, i.e. noise.

## What I would try with more budget

The binding constraint was *information*, not compute: with only calendar, clock, carrier, airport, route
and distance, the model sits near its ceiling and every remaining calendar- or identity-based feature
encodes year-specific noise. The most valuable next step would be a **time-ordered cross-validation scheme
inside the training year** (fit on early months of 2005, validate on later months) used to *screen* features
and capacities, so that candidates are judged on cross-time generalization rather than on `eval.csv`; that
would have caught the calendar and target-encoding failures immediately instead of experiment by experiment.
After that: stacking a second-level XGBoost on out-of-fold predictions of the base bag (the bag members are
correlated but not identical, so a learned blend could beat the flat average), scaling the bag to 30–50
models now that `max_bin=128` makes each one cheap, and a joint sweep over `max_bin` / `colsample_bytree` /
`max_depth` — the `max_bin` result was found only in the last two experiments and the interaction between
bin granularity and column subsampling is clearly not fully explored.
