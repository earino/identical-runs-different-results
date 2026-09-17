# Final report — airline delay XGBoost

**Best Eval AUC: 0.7275** (baseline 0.7141, +0.0134). HEAD (`db8312b`) is the final `train.py`;
the last experiment re-ran it end-to-end and reproduced 0.7275, and `validate.sh` prints `CONTRACT OK`.

## Changes that mattered most

1. **Strong regularization for temporal shift** (E7-E8, +0.005): train is 2005, eval/holdout 2006. Internal
   2005-validation AUC (~0.75) far exceeds 2006 AUC (~0.71), so variance reduction transfers: max_depth 6→4,
   min_child_weight→40, gamma 2, reg_lambda 10 was worth ~+0.005 AUC. This was the biggest single lever.
2. **Carrier × hour-of-day crossed categorical** (E15, +0.007): 480 levels capturing carrier scheduling
   patterns (hub banks, red-eye structure) that are stable across years and too deep a interaction for
   regularized trees to find on their own. The single largest feature-engineering win.
3. **Early stopping tuned for transfer** (E31-E33, +0.001): patience 25 and a 20% stratified train split stop
   earlier and estimate `best_n` more reliably — fewer trees generalize better under the year shift.
4. **3-model ensemble** (E27-E28, +0.001): average of depth 4/5/6 models (seeds 42/7/3), each early-stopped
   then refit on the full train. 5-6 models did not help further.
5. **Stable calendar features** (E5, +0.001): day-of-year, weekend flag, red-eye flag. Modest but ablation
   confirmed they earn their place (+0.0026 when removed at E34).

## What did not help

1. **Route (Origin_Dest) as a categorical** — 4-5k sparse levels; overfits 2005 route patterns, −0.013 (E4).
   Even heavy-smoothed route target encoding hurt (−0.005, E26); route delay propensity does not transfer.
2. **Airport × hour crossings** (E16, −0.009): 6.8k levels each, too sparse; carrier×hour was the sweet spot.
3. **Target/frequency encodings of Origin/Dest/carrier**: smoothed TE was neutral pre-carrier_hour (E6),
   mildly positive after (+0.002, kept); frequency encoding hurt (E23); distance-normalized-by-carrier hurt
   (E22); dow×hour and carrier×dow crossings hurt (E18).

## With more budget

I would (a) extend the interaction family more carefully — e.g., carrier×hour with coarser hour bins or
carrier×(hour×season) with shrinkage toward the carrier×hour mean, since finer crossings kept overfitting;
(b) try a stacked second-level XGBoost over out-of-fold predictions of the 3-model ensemble; (c) run a small
Bayesian search jointly over (depth, mcw, gamma, lambda, lr, patience) under the carrier_hour feature set —
the manual path found two flat plateaus (0.7187, 0.7275) suggesting a joint optimum nearby; and (d) probe
whether per-feature monotone constraints on DepTime/distance improve cross-year stability.
