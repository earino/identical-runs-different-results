# Final report — airline dep_delayed_15min

Best Eval AUC: **0.7527** (baseline 0.7141, +0.0386). 40/40 experiments used.

## The 5 changes that mattered most

1. **OOF target encoding of high-cardinality categoricals** (carrier/origin/dest, route=Origin_x_Dest): +0.002 over baseline-era features, but it was the *foundation* every later gain built on. 5-fold OOF at train time, full-train refit for eval/holdout.
2. **Fine-grained time-interaction TEs** — route/origin/dest/carrier crossed with 30/15/10-minute departure-time buckets: the single biggest jump, 0.7249 -> 0.7387 -> 0.7410 as granularity refined. Hour-of-day is the dominant delay driver and the interactions capture terminal/congestion effects that shift by time of day.
3. **Heavy TE smoothing (m=600, not 20)**: 0.7410 -> 0.7441. With 100k rows and thousands of fine-grained keys, strong shrinkage toward the prior turns noisy cell means into stable empirical-Bayes features; the optimum was two orders of magnitude above textbook defaults.
4. **Logit-space ensemble of three decorrelated XGB views** (full raw+TE / raw-categorical-only / TE-only), weights 40/35/25: 0.7441 -> 0.7519. Different feature views err differently; simple averaging of logits captured it.
5. **Time-decay sample weights** (linear 0.35->1.0 across 2005 months, all models): 0.7519 -> 0.7524. Eval and holdout are 2006, so later training months are more relevant to the distribution shift.

## 3 things that did not help

- **Congestion count features** (flights per origin/dest/day/hour computed on the train slice): 0.7249 -> 0.7223. Counts from a 100k-row sample are a biased proxy for true airport traffic and don't transfer across years.
- **Day-grouped OOF folds for the TE** (encode each row from other calendar days): 0.7524 -> 0.7276, catastrophic. The fine-grained TE signal is *within-day*; removing same-day rows starves exactly the cells that carry it.
- **Replacing unseen-key NaN with the global prior / month-interaction TEs / 5-model and 2-seed bagged ensembles**: all neutral-to-negative (0.7430, 0.7507, 0.7511, 0.7522). XGB handles NaN natively and extra same-view members only diluted the three views that worked; month interactions overfit 2005 seasonality.

## What I would try with more budget

The TE family is the engine but it is still a single "one key = one prior" encoding. I would next build *hierarchical* target encodings — shrink each fine cell not toward the global prior but toward its parent key (route_x_t10 shrunk toward route_x_hour toward route), which should recover signal in sparse cells that m=600 currently flattens; that is the most likely source of another +0.005. Second, the ensemble weights were hand-tuned on eval; fitting a stacked XGB (or simple weight search) on out-of-fold predictions would pick them more honestly. Third, I would revisit granularity below 10 minutes (per-minute TE on DepTime) with hierarchical smoothing, and finally I would test a 2005-holdback validation split (train Jan-Oct, validate Nov-Dec) to get a time-honest model-selection signal instead of leaning on eval.csv — everything above was selected on a single 100k-row 2006 slice, so small gains (<0.0005) are within selection noise.
