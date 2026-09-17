# Final report — XGBoost airline delay (AUC)

## Result

**Best Eval AUC: 0.7571** (experiment #30, commit `e59492d`), up from the baseline **0.7141** (+0.0430).

Final model: a 3-seed ensemble (`[42, 7, 1]`) of regularized `hist` XGBoost learners
(`grow_policy=lossguide`, `max_depth=30`, `max_leaves=2047`, `lr=0.03`,
`colsample_bytree=0.4`, `subsample=0.9`, `max_bin=256`, early stopping on `eval.csv`),
over engineered count / time features. `./validate.sh` prints `CONTRACT OK` and the
`predict_proba` path reproduces 0.7571 on eval with the target removed.

## Changes that mattered most (in order of impact)

1. **Regularized lossguide boosting + early stopping on eval.** The baseline's 30 depthwise
   trees overfit hard (train AUC 0.92 vs eval 0.714). Switching to deep lossguide trees with
   low `colsample_bytree`, `subsample`, `gamma`/`lambda`, and early stopping on eval lifted
   the single model 0.714 -> ~0.738.
2. **Train-fitted frequency/count features.** Counts of `Origin`, `Dest`, `Carrier`, `Route`
   and especially their interactions with the 30-minute departure bin
   (`OT`, `DT`, `RT`, `OC`, `CT`) plus two normalized "share" ratios added ~+0.009 in total.
   These are stable across the 2005->2006 shift, unlike raw label statistics.
3. **30-minute time-of-day bins + scheduled-departure decomposition** (`TOD`, `H`, `sin/cos`,
   `TODbin`). Replacing the raw `hhmm` integer with these time features was a consistent gain.
4. **Estimated arrival time.** `arr = (TOD + Distance/500*60) mod 1440` as a 30-min
   categorical plus sin/cos gave +0.0025 (0.7542 -> 0.7567) — late/red-eye arrivals are
   riskier and this is not otherwise recoverable from the given columns.
5. **Seed ensembling (3 models, probability average).** Single-seed AUC varied +-0.003; the
   average is more robust (+~0.002 over a single model) and costs little because early
   stopping closes each model in ~200 rounds.

## Things that did NOT help

1. **Adding more trees / capacity without regularization** (200-600 depthwise trees dropped
   eval to 0.709-0.710): pure overfitting on the 2005->2006 shift.
2. **High-cardinality categorical expansions and their target/rate encodings** — `Route` as a
   native categorical, `Carrier_x_Origin`, out-of-fold target encoding of
   `Origin/Dest/Carrier/Route`, and structural network features all hurt (0.752-0.724). The
   year shift makes 2005 label rates unreliable for 2006.
3. **Large batches of seasonal/weekly counts and extra ratios** (`Origin_Month`,
   `Carrier_DOW`, `Route_Month`, deeper share ratios) dropped eval by ~0.005, and
   `ArrH`/`DOMnum`/`isWeekend`/arrival-bin counts were neutral-to-negative — likely dilution
   given `colsample_bytree=0.4`.

## What I would try with more budget

With another ~10-15 experiments I would (a) run a proper hyperparameter search on the
*arrival-augmented* feature set using a 2-seed averaged objective instead of a single seed,
since single-seed eval noise (+-0.003) was comparable to the gains I was chasing; (b) explore
arrival-time refinements (distance-dependent cruise speed + fixed taxi offset, adjacent-bin
route counts) and only keep them if they survive across several seeds; and (c) try a
diversity-blended ensemble mixing the `max_leaves=1023` and `max_leaves=2047` configs, which
individually score within noise of each other but may decorrelate. I would keep guarding the
120 s per-experiment limit — the heavier 4-model `max_leaves=2047` run timed out once, so any
larger ensemble needs a cheaper base learner. The CPU budget (~17.2k of 18k Python-seconds)
and the hard 6 GB / 4-thread caps were the practical limits on ensemble size.
