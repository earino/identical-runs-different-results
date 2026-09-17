# Final report — airline delay XGBoost (autoresearch, scenario 2)

**Best Eval AUC: 0.7289** (baseline: 0.7141, +0.0148)

Final `train.py`: 20-model seed bag of XGBoost (hist) on calendar/time features,
colsample_bynode=0.2, depth 14, ~50 rounds chosen by a temporal early-stopping probe
(fit months 1-10 of 2005, validate Nov-Dec 2005), learning rate 0.1, subsample 0.8.

## The changes that mattered most

1. **Seed bagging** (20 XGB models, averaged probabilities): 0.7141 -> 0.7177. The single biggest, most robust win.
2. **Aggressive colsample_bynode + deeper trees**: bynode 0.2/0.3 with max_depth 8 -> 12 -> 14 unlocked a compounding ladder: 0.7211 -> 0.7244 -> 0.7280 -> 0.7289. Feature-limited splits let deep trees act like implicit random-forest members, boosting both individual and ensemble quality.
3. **max_bin=512/1024** for finer time-of-day splits: +0.001-0.0012 on its own.
4. **Time features**: cyclic tod (sin/cos), hour, cyclic day-of-year (seasonality): ~+0.003 combined.
5. **Temporal early stopping** (validate on Nov-Dec 2005, refit bag on all data with chosen rounds): +0.0004 and a principled, shift-matched way to pick rounds.

## Things that did NOT help

1. **Target encoding** of Route/Origin/Dest/Carrier (smoothed, OOF): 0.7080 — 2005 level statistics don't transfer to 2006; TE encourages memorizing year-specific rates.
2. **Route as a native categorical** (700+ levels): 0.7027 — partition-on-category splits overfit routes.
3. **More capacity via plain boosting** (150-400 rounds, lower lr, heavier tree depth without bynode sampling): never beat ~50-60 rounds; the 2005->2006 shift punishes variance.
4. Frequency encoding, holiday flags, weekend/summer interactions, recency weighting, cyclic dow/month, one-hot partitioning (max_cat_to_onehot), dropping the carrier, depth-mix bags: all neutral or worse.

## What I would try with more budget

The two productive axes suggest an ensemble of "deep trees + heavy feature sampling" is well-suited to
shifted-year airline data, so I would push further along the random-forest-in-XGB direction: even
deeper trees (max_leaves with lossguide growth), bynode 0.1-0.15, hundreds of rounds with ES chosen on
multiple temporal folds (months 7-8, 9-10, 11-12), and 50-100 bag members — plus a second bag trained
on day-level bootstrap samples for row-diversity. I would also test interactions of carrier/hub with
hour through explicit per-carrier departure-hour histograms computed WITHOUT labels (schedule shape,
not delay rates), and a small rank:pairwise-tuned member to sharpen the ranking directly. Finally I
would re-verify the top-3 configs across several seeds to separate <0.0005 noise from real gains
before committing to them.
