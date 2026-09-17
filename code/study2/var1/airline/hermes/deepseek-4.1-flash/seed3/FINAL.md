# Final report — airline delay classification (XGBoost, AUC)

**Best Eval AUC: 0.7393** (experiment #39, commit `7605145`) vs **0.7141** for the unmodified baseline.
40/40 experiments used, ~14450 of 18000 CPU-seconds, ~103 s per final run (cap 120 s). `./validate.sh`
prints `CONTRACT OK` and reproduces the same AUC through `predict_proba(df)` with the target column removed.

## Changes that mattered most

1. **Time-of-day × carrier feature engineering** (0.7141 → 0.7263, exp #6). `DepTime` (hhmm) was turned into
   hour, minutes-since-midnight and cyclical sin/cos terms, plus a `carrier_hour` categorical interaction,
   `log1p(Distance)`, and train-fitted route/carrier frequency counts. Hour-of-day is the dominant delay
   signal and each carrier has its own intraday congestion profile; all of it lives inside `prepare()`, so it
   also applies to unseen rows.
2. **Bagged ensembles of shallow models** (0.7263 → 0.7276, exp #7). A single model is high-variance and the
   2005 → 2006 shift punishes capacity, so several depth-3/4/5 members are averaged rather than one bigger tree
   ensemble being grown.
3. **Random feature subspaces per member** (0.7301 → 0.7331, exps #11–#22). Each member sees a randomly sized
   subset of the columns; this decorrelates members far more than seed changes and was worth ~+0.004. Uniform
   70 % subspaces gave +0.002 (exp #11), random-size subspaces another +0.001 (exp #17), and 36 members another
   +0.001 (exp #22).
4. **Per-member hyperparameter jitter** (0.7331 → 0.7386, exps #28–#31). Depth (2–10), learning rate,
   `reg_alpha`/`reg_lambda`, `min_child_weight`, `subsample`, `colsample_bytree`, `max_cat_threshold` and the
   categorical encoding mode (one-hot vs partition) are all drawn independently per member. This was the single
   biggest late gain: a 36-member pool with heterogeneous members beats a homogeneous one.
5. **Logit-space averaging** (0.7386 → 0.7393, exp #39). Averaging member logits instead of raw probabilities
   nudged the AUC up; weighting that mean by subspace size was a tie and was dropped for simplicity.

## Things that did not help (and were reverted)

1. **More capacity.** 500 trees / depth 7 scored 0.7055 against the 30-tree baseline's 0.7141, and early
   stopping on an internal 20 % split scored 0.7121. Train-internal CV keeps improving with capacity
   (0.743 → 0.766) while eval AUC falls — pure overfitting to 2005.
2. **Target encoding of Origin_Dest.** CV rose to 0.757 but eval AUC collapsed to 0.699: route delay rates do
   not transfer across the year boundary, so train-fitted statistics actively hurt.
3. **Extra categorical structure** — route (Origin_Dest) categorical, origin/destination × hour interactions,
   month × carrier, distance bins, calendar ablations, per-member row bagging, 6-column subspaces, an OOF
   XGBoost stacker over member predictions (0.7275), rank averaging and subspace-size weighting. None beat the
   corresponding baseline; several cost 0.003–0.02 AUC.

## What I would try with more budget

The remaining gap is variance, not bias. Two ensembles built with the same configuration but different random
draws measured 0.7386 and 0.7344, so the eval-side noise floor is roughly ±0.002 and most late keep/revert
decisions sat inside it. With a bigger compute envelope the highest-value move would be to stop tuning against
eval.csv at all: build a much larger pool (150+ members) with the same jittered/subspaced recipe, average it in
logit space, and pick hyperparameters by repeated random draws rather than a single eval number, which would
remove the draw-level luck from the final model. After that I would revisit DART (clearly under-trained at the
120 s limit, but its eval AUC was within 0.001 of the bagged pool at ~5× the cost), an OOF-weighted blend whose
weights come from out-of-fold AUC instead of uniform averaging, and — since calendar columns (Month,
DayofMonth, DayOfWeek) transfer almost nothing across the year boundary — a systematic re-test of which
features survive a 2005→2006 style shift, using the last months of train as a shift-validation fold.
