# Final report — airline dep-delay AUC (autoresearch XGBoost)

## Best result

- **Eval AUC: 0.7506** (baseline: 0.7141, +0.0365)
- Best commit: `4f8b24e` ("ratio features for 30m and 2h windows") — HEAD of branch `experiment`.
- Validation: `./validate.sh` → `CONTRACT OK` (predict_proba reproduces 0.7506 with the target column removed).
- 40/40 experiments used; 150 minutes left on the wall clock (budget hit the experiment cap first).

## The 5 changes that mattered most

1. **Capacity**: 30 trees → ~1000 trees at depth 8–12, lr 0.05 (0.7141 → 0.7247). The baseline was badly underfit.
2. **Departure-time features**: DepHour/DepMin/DepMinutes + sin/cos encodings of minute-of-day, plus numeric + cyclical Month/DayofMonth/DayOfWeek (0.7169 at 300 trees; sin/cos ablation later cost −0.0018, confirming they matter).
3. **Volume/count encoding family** (fit on train only, no target): counts of flights per Carrier/Origin/Dest/Route and per key×hour (OriginHour, DestHour, CarrierHour, OriginHourDOW, RouteHour, CarrierRoute, CarrierOriginHour, MonthOriginHour, ...). Each successive key×hour count added AUC; this family was the single biggest engine of gains.
4. **Queue features**: rolling-window counts of departures at the origin (and estimated arrivals at the dest) within ±30min / ±1h / ±2h of this flight's scheduled time, from train-fitted airport×48-bucket matrices — a physically-motivated congestion signal (0.7378 → 0.7396).
5. **Ratios and interactions**: queue ÷ hour-typical volume ("unusually busy for this hour"), queue×hour, distance×hour products, and colsample-diverse 3-model XGB ensemble with min_child_weight=1 (0.7396 → 0.7506 in steps).

## 3 things that did not help

1. **Target encoding** of Carrier/Origin/Dest/Route — hurt both as replacement (0.7107) and alongside (0.7119): train(2005)→eval(2006) delay-rate shift makes smoothed target means unreliable per level.
2. **Route as a native categorical** (4198 levels on 100k rows): overfit, 0.7193. And 5-model/row-bagged ensembles: 5 models blew the 120s cap; 80% row bags lost more to data than they gained in diversity (0.7363).
3. **Extra regularization / log1p counts / cumulative-day-position / gap-minutes / per-model round counts / 5-seed averaging** — all neutral to worse (0.738–0.7505). The winner's configuration was already near a local optimum; only hour-typical *ratio* features added anything after exp #29.

## What I would try with more budget

The biggest unexplored direction is **same-day sequential structure**: the dataset has no flight-id/tail-number, but with more time I would reconstruct per-airport, per-day departure sequences from (Month, DayofMonth, DepTime) and compute propagated-delay proxies — e.g. the delay rate of the k-1 flights scheduled before this one at the same origin (a lagged target statistic with proper time-ordering, which is leak-free on the holdout but expensive to validate). Second, per-key *rate* features (delayed fraction per Origin×Hour with heavy smoothing and shrinkage toward the parent Origin rate, hierarchical/empirical-Bayes) may succeed where flat target encoding failed, because the hierarchy pools sparse keys. Third, a two-stage model — predict arrival-side queue pressure at the origin's connecting airports — and finally smarter ensembling: rank-average over depth-diverse models was equal to probability mean here, but 10+ diverse models (varying colsample, max_bin, and feature subsets) would likely add +0.001–0.002 with the compute to fit them.

## Reproduction

- `train.py` at HEAD trains in ~112s, evaluates in ~4s, prints `Eval AUC: 0.7506`.
- Full log: `experiments.tsv` (40 rows), one commit per experiment on branch `experiment`.
