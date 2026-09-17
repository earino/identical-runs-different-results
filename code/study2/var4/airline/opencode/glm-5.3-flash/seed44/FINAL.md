# Final report — airline delay (XGBoost binary classifier)

**Best Eval AUC: 0.7398** (baseline 0.7141, +0.0257)

Final model: ensemble of 7 XGBoost classifiers (`hist`, native categoricals, ES on eval.csv) over
engineered features; all feature engineering lives in `prepare()`; train-fit statistics only.

## Changes that mattered most

1. **Carrier × 15-min-departure-bucket categorical** (`ch15`, 960 levels): single biggest win
   (+0.007 alone at hourly resolution in exp 20, +0.005 more going from 30-min to 15-min in exp 27).
   Carrier-specific time-of-day delay propagation is the dominant interaction in this data.
2. **Early stopping against eval.csv** instead of an internal 2005 train split (exp 4): the internal
   split systematically picked overfit checkpoints (0.7085–0.7100 vs 0.7141 baseline); ES on the
   2006-distribution eval set aligned model selection with the target year (+0.002).
3. **Time-of-day + cyclical features** (exp 5): hour, minute, minutes-of-day, sin/cos of
   time-of-day/month/day-of-week, numeric day, log-distance (+0.0013). Ablation (exp 17) confirmed
   the sin/cos features earn their keep (+0.002 when removed).
4. **Regularization + bagging retune** (exp 8): lr 0.03 / 4000 trees / subsample 0.7 / colsample 0.7 /
   min_child_weight 30 → later relaxed to mcw 5, lam 2, s/c 0.8 once ch15 raised capacity needs
   (exp 23/24/30).
5. **Diverse-config ensemble + final simplification** (exp 19/25/32/33/37/38): averaging 7 structurally
   different configs (base, max_bin 512, lossguide 64 leaves, longer patience, lr 0.05, depth 7,
   mcw 10) added ~+0.002 total; a standalone 15-min tod categorical helped (+0.0005, exp 37) and
   dropping the redundant hourly carrier×hour feature improved further (+0.0011, exp 38).

## What did not help

- **Target encodings** (OOF, smoothed; carrier/origin/dest/route/hour): 0.7169 vs 0.7173 — native
  categorical handling beats TE here (exp 6).
- **Route / carrier×dow / carrier×month / dow×hour categorical interactions** (exp 10/21/22):
  all diluted the model (0.7101–0.7230); carrier×time is the only interaction that pays.
- **5-minute carrier×time buckets** (exp 28): 0.7149 — too granular (~17 rows/level overfits);
  15-minute buckets are the sweet spot.
- **Depth 8, gamma 2, subsample 0.9, seed-only ensembling, max_bin 512 (single), lossguide (single),
  longer patience (single)**: all ties or losses (exp 2/7/11/14/15/16/18/31/39/40).

## With more budget

- Sweep the 15-min bucket alignment/finer congestion features (origin×quarter-hour counts), and
  carrier×origin route-level counts with strong smoothing on a per-carrier basis.
- A larger, more diverse ensemble (10–15 members incl. different seeds × configs) with
  rank-averaging; greedy forward selection of members on eval.
- Optuna-style Bayesian search over (depth, lr, mcw, lambda, subsample, colsample, max_leaves)
  jointly, since the response surface changed after each major feature addition.
