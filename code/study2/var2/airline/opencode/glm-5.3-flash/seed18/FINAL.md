# Final report — airline delay AUC

**Best Eval AUC: 0.7524** (baseline 0.7141 → +0.0383 over 40 experiments). Final config: 2-seed XGBoost
ensemble; each member = ES probe (lr 0.02, 30% random val, patience 200) to pick the tree count, then a
refit on 100% of train with that fixed count; predictions averaged.

## What mattered most

1. **Unbounded-depth trees + early stopping** (max_depth=0, lr 0.01→0.02, ES on a random 20–30% val split):
   +0.017 over the depth-6 baseline. Deep trees find the hour×carrier×route interactions; ES controls overfit.
2. **Probe-then-refit**: run ES on 70% of rows, then refit on 100% with the ES-chosen tree count (+0.004).
   More data per tree at the same effective complexity.
3. **Strong stochastic regularization**: subsample 0.7, colsample_bytree 0.6, colsample_bynode 0.5 (+0.011
   vs 0.9/0.9/none). The single biggest hyperparameter win; deeper trees need heavier sampling.
4. **Feature engineering**: parse c-N strings to ints; DepTime → hour/minute + cyclical sin/cos; log
   distance; log1p count encodings (train-fit) for carrier/origin/dest/route (+0.004 combined).
5. **Seed-averaged refit ensemble** with per-member colsample_bynode diversity (0.5/0.6), sharing one probe
   to fit the 120 s cap (+0.002). Holiday-travel window flags (winter, Thanksgiving) added +0.0006.

## What did not help

1. Out-of-fold target encoding (carrier/origin/dest/route, m=20) — native categorical splits already capture
   that signal; TE added overfit noise (−0.002).
2. Leaf-wise growth (lossguide, max_leaves=256) and explicit cross categoricals (carrier×hour, dow×hour).
3. Cyclical dom/dow encodings, gamma=1, reg_lambda=5, min_child_weight=10, a temporal (month 10–12) ES
   split, and Month/DayOfWeek as native categoricals instead of ordinal ints.

## With more budget

Three-plus lr-diverse refit members (shared probe keeps it under the cap), a DART member, a finer 2-D grid
around (subsample, colsample_bytree/bynode), max_bin tuning, and per-member feature subsets. I would also
re-check ES iteration choice by averaging AUC over two validation folds rather than one.
