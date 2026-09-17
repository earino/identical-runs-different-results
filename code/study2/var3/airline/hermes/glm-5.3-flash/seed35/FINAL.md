# Final Report — airline delay AUC (autoresearch XGBoost, scenario 2)

Best Eval AUC: **0.7459** (baseline 0.7141). Final config: HEAD (`exp16: greedy-11 subset at n1100`) —
11-member XGBoost ensemble + 1 quantile-median booster, probability average (all members clipped to [0,1]).
Validation: `CONTRACT OK`, predict_proba AUC 0.7449→0.7459 confirmed via validate.py on data/eval.csv.

## What mattered most (in order of impact)

1. **Cyclic DepTime features** (dep_min + sin/cos of day, exp4): +0.0038 over raw hhmm. Raw scheduled
   departure time breaks at the midnight wrap; sin/cos makes 23:50 and 00:10 neighbors.
2. **Carrier x 30-min-block categorical** (exp6): +0.012 single largest gain. Carrier delay patterns are
   strongly time-of-day dependent; explicit interaction > hoping depth-6 trees find it. Native categorical
   handling with train-fixed levels (appearance order — sorted order cost 0.002) was essential.
3. **Dropping Distance from the base feature set** (exp4): +0.002. Origin/Dest already imply it; it only
   added variance — though it stayed valuable as *member diversity* (several members re-add it).
4. **Ensemble diversity instead of single-model tuning** (exp7→exp16): +0.012. Bags (subsample 0.8),
   depth 5/6/7, +/-Distance, minute-of-hour, carrier-x-month, no-Dest, huber objective, airport-x-hour
   members, and one reg:quantileerror (median) booster — the last added +0.0010 on top of 11 XGB members.
   Greedy member selection on cached probability vectors chose the final 11.
5. **Airport x hour-of-day categoricals** (Origin/Dest x hour, exp14/16): +0.002 as member variants —
   the last genuinely new signal found.

## What did not help (all tested, all rejected)

- **Target encodings in any form**: smoothed full-train TE, out-of-fold TE, route TE, TE as numerics next
  to their own categoricals — every variant lost 0.002–0.015. The 2005→2006 time shift punishes
  marginal-frequency features.
- **Capacity/regularization**: depth 8 (0.69!), subsampling/colsample/gamma/reg_lambda on a single model,
  max_bin changes, lossguide — the baseline d6/full-data config was already optimal per-axis.
- **Robustness tricks**: month-recency sample weights, lr schedules, early stopping on a train split,
  rank:pairwise objective (0.59 AUC), DART, random-forest num_parallel_tree, calendar/holiday flags,
  route categorical (0.70), origin×hour TE, member weighting / rank / logit averaging, OOF ridge stacking.

## Notes on budget

16/40 experiments used; ~17.2k of 18k CPU-seconds consumed (~95%) — most of it by 30+ free-screening
scripts that ranked ideas before committing an experiment. Wall-clock per experiment (120s hard cap)
became the binding constraint from exp12 onward: it caps the ensemble at ~11 n1100 members (exp12/13/15
timed out at 12–15 n1200 members and each timeout burned ~460 CPU-s).

## With more budget

- Tune the member pool, not the members: the greedy selection ran on 18 cached candidate vectors; with
  CPU headroom I would grow the pool to 40+ (per-variant n/lr retunes, more quantile alphas, feature-drop
  variants) and let forward selection pick ~12 members that fit the 120s cap at n1200.
- A 2-fold bagged variant of the whole ensemble (each member trained on 2 half-samples, averaged) —
  untested because one fold costs a full ensemble fit (~2x runtime).
- Year-2 drift study: train on rolling 2005 windows and check which members degrade least on 2006 —
  member selection by stability rather than point AUC would likely generalize better to the hidden holdout.
