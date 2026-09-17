# FINAL — autoresearch XGBoost (airline delay, 2005 -> 2006)

**Best Eval AUC: 0.7556** (commit `3cfd021`, experiment #35 of 40, 93 s per run)
Baseline was **0.7141** (unmodified `train.py`, experiment #1) — a gain of **+0.0415 AUC**.
Full log: `experiments.tsv`. `./validate.sh` prints `CONTRACT OK` (0.7556 reproduced through
`predict_proba` with the target column removed), so all feature engineering lives inside `prepare()`.

## Changes that mattered most

1. **Deleted the raw identity categoricals.** `Origin`/`Dest` (300 levels), `DayofMonth`, `Month`,
   `DayOfWeek` were originally fed to XGBoost as native categoricals. Partition splits on those memorise
   one year's quirks and do not carry to 2006. Dropping them (`+0.0049`, `+0.0071`, `+0.0014` for
   Origin/Dest, DayofMonth and Month respectively, the last replaced by numeric + cyclical month) was the
   single biggest win, and it *unlocked* the deep trees below: once the model can no longer memorise
   airport codes, extra depth pays off instead of overfitting.

2. **Airport/carrier "wave" profile features.** For each flight, the hourly traffic volume of its origin
   airport (departures and arrivals), of its destination, and of its own carrier, read at lags 1–12 and
   leads 1–12 from the train year. Each is a single `Series.map` lookup on a `(key|hour)` dictionary
   fitted on training data only. Gains came in steps: lag-1 (+0.0031), lags 1–3 (+0.0006), 1–6 (+0.0003),
   1–12 (+0.0010), leads 1–12 (+0.0006), carrier wave (+0.0008). Dropping the destination side cost
   −0.0020, so the destination traffic really is signal.

3. **Peak-hour congestion counts** (origin/dest/carrier × departure hour) plus the count of aircraft
   *arriving* at the origin in that hour (late-inbound propagation): +0.0034, +0.0005.

4. **Departure-time decomposition.** hhmm-as-integer has broken geometry (1357 → 1400 spans 43 units for
   3 minutes); dropping the raw column for `dep_hour`/`dep_min`/`dep_tod` was +0.0011 on its own.

5. **Deep bagged ensemble.** Six XGBoost members with differing depths (8–14), subsampling, column
   sampling and seeds, averaged. Diversity first paid (+0.0015 over a single model at depths 5–7), and
   after the identity features were removed the optimal depth jumped: depths 6–12 (+0.0054) and 8–14
   (+0.0036) each added far more than depth ever did in the old feature space.

## Things that did not help (reverted)

1. **Route (`Origin_Dest`, ~2500 levels) as a categorical: −0.0170.** The clearest overfitting signal in
   the whole run, and the result that motivated the identity-feature purge above.
2. **Smoothed target encoding** of origin/dest/carrier and their (×hour) keys, k=20: −0.0018. The
   year-to-year drift of 2005 delay rates outweighs their value in 2006.
3. **Regularisation of the deep ensemble** (`min_child_weight=4`, `lambda=1`): −0.0026; a leaf-size floor
   is actively harmful here, and single-model sweeps of `max_depth=10`, 1500 trees and a
   slower/regularised lr-0.03 config were all −0.002 to −0.003 when the identity features were still in.
4. Also flat (~0.000): calendar-conditional and carrier-mix counts, plain per-entity counts without the
   hour interaction, congestion *shares*, holiday-proximity features, and seed-only bagging (with
   `subsample=1.0` XGBoost is nearly deterministic, so seed variation alone changes nothing).

## What I would try with more budget

The largest remaining lever is capacity versus the 120 s per-experiment ceiling: the winning ensemble
already runs 93 s, and a depth-16 variant scored 0.7560 at 109 s but was discarded as too close to the
timeout. With more budget I would first buy back time — `max_bin` reduction, int8/float32 downcasts for
the ~110 engineered columns, and profiling which members actually contribute — then spend it on a wider
deep ensemble plus a `lossguide`/`max_leaves` member, which is a more token-efficient way to spend depth
than `max_depth`. Second, I would replace the single 2005→2006 eval signal with an inner time-ordered
split (fit on early-2005 months, validate on later ones) so that depth, rounds and ensemble weights can
be chosen without touching `eval.csv`; the final ten experiments were decided on differences of
0.0004–0.0005, which is at the noise floor of a 100k-row AUC, and several of those decisions would be
better made on repeated inner folds. Third, now that identity features are gone, I would revisit the
*smooth* forms of the ideas that failed in the old space — a shrunk, cross-fitted airport delay rate
rather than the plain mean, and route-level congestion expressed as a count/rank instead of a
categorical — since the −0.0018 and −0.0170 results were obtained with those 300-level categoricals
still competing for splits. Finally, I would extend the traffic-profile idea one level deeper into
interactions (origin congestion × this carrier's share of that hour, connection-wave structure) and check
whether the profile counts transfer better when normalised per airport, as the lags currently conflate
airport size with congestion.
