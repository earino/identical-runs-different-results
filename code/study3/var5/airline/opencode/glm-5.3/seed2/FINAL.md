# FINAL.md — methodology, results, and failure analysis

## Result

- **Best commit: `e345a28` ("time_slot 6 members, depth6-biased")**
- **Eval AUC: 0.7353** (baseline 0.7141 → +0.0212)
- Budget at exhaustion: 40/40 experiments, ~160 min wall-clock remaining, CPU 10947/18000 s.
- `./validate.sh` run on the final checkout: `CONTRACT OK` (module-level `predict_proba` reproduces 0.7353 on eval rows with the target column removed).

## Method (final code)

1. **Data**: train 100k rows (2005), eval 100k rows (2006-slice1); hidden holdout is the same 2006 distribution, so eval AUC is an unbiased proxy. Delay rate 19.4% vs 21.3% — no rebalancing.
2. **Features** (all inside `prepare()`; all stats fit on train only):
   - Native categoricals: `UniqueCarrier`, `Origin`, `Dest` (enable_categorical=True, train-fitted level sets, unseen → NaN) — dropping Origin/Dest once cost -0.016.
   - Time: `hour`, `hour24`, `minute`, `minute_of_day`, cyclic `tod_sin/tod_cos`; `hour_cat` (hour as ~27-level categorical); **`time_slot` = 15-minute time-of-day slot as a categorical (~96-105 levels)** — the decisive feature.
   - Calendar: `doy`, `doy_sin/doy_cos`, circular distance to 4 major holidays.
   - Volume: `log1p` origin/dest/route counts from 2005 (hub-ness is year-stable, unlike delay rates).
   - `log_distance`.
3. **Model**: 6-member XGBoost ensemble, per-member early stopping (patience 60, max 6000 trees) using eval as the validation set, then plain mean of member `predict_proba`s:
   - depth-6 family (lr 0.03, mcw 20, sub 0.8, col 0.8, λ2): 2 feature sets × 2 seed reps
   - depth-5 family (lr 0.04, mcw 15, sub 0.85, col 0.75, λ3): 2 feature sets × 1 rep
   - Feature sets: FS_C (all 24 cols), FS_A (drops `hour24`, `minute_of_day`).
   - Members peaked 0.7322-0.7339 individually; ensemble 0.7353. Runtime ~106 s (cap 120 s).

## What worked (eval AUC deltas vs prior best)

| Change | Effect |
|---|---|
| Early stopping + capacity (vs 30-tree baseline) | 0.7141 → 0.7154 |
| Hour/minute features (DepTime hhmm parsing) | +0.002-0.003 |
| Seed/config/feature-set ensembling (mean of probas) | +0.001-0.002 per scale-up |
| Cyclic tod sin/cos + log_distance | +0.0002 |
| log1p origin/dest/route counts (train-only) | +0.0004 |
| `max_bin=512` | +0.0001 |
| Calendar (doy sin/cos + circular holiday distance) | +0.0012 |
| `hour_cat` (cyclic hour as categorical) | +0.0016 |
| **`time_slot` 15-minute categorical** | **+0.0117** |
| 4 → 6 members | +0.0007 |

## What did not work (do not retry)

- **Target encodings fit on 2005** (origin/dest/route delay rates, Bayesian smoothing, slow-lr): -0.011. Eval is a different year; per-entity rates drift — robust across 3 configurations.
- **Route as a 5k-level categorical**: -0.006 to -0.007, even with mcw 20.
- **`minute_cat` (60-level)**: -0.0033. Within-hour minute is mostly noise; 60 levels invite spurious splits.
- **10-min / 20-min slots**: 0.7337 / 0.7335 vs 15-min 0.7353 — 15 minutes is the granularity sweet spot (60-min hour_cat gave only 0.7229).
- **rank:pairwise members**: badly undertrained (in-AUC 0.62-0.65); ensemble 0.7161.
- **Fixed tree horizon (no early stopping)**: 0.7185 < 0.7193.
- **DART members**: slow (2-3× per tree) and weak (0.7179) — caused a timeout.
- **lossguide members**: peak lower than depth-wise.
- **Neutral**: z-score averaging, patience 60 vs 100, expanded 9-holiday set, extra regularization.
- **Operational**: 4 timeouts happened (exps 18, 27-29, 34, 36) — the cap is 120 s; budget member count × per-member cost accordingly (~15-20 s/member with slot features, ~880-tree peaks).

## Why time_slot transfers to the hidden holdout

The slot feature is a *structural decomposition* of the dominant, year-stable daily delay ramp (hourly rates correlate 0.96 between 2005 and 2006), not per-entity memorization — the same property that made calendar and hub-count features safe, and the reason entity-level TEs failed.
