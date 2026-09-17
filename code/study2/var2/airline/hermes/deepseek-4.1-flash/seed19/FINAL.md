# FINAL — airline dep_delayed_15min, XGBoost

**Best Eval AUC: 0.7479** (9-member XGBoost ensemble, commit `c0af5f7`) vs. **0.7141** baseline.
40/40 experiments used, ~3245 of 18000 CPU-seconds, every run well under the 120 s cap (max 81 s).
`./validate.sh` → `CONTRACT OK` (predict_proba recomputed 0.7479 with the target column removed).

## What mattered most

1. **Group target-rate encodings cross-fitted out-of-fold (5-fold), plus log traffic counts.**
   ~20 keys (Origin, Dest, carrier, route, and their products with time-of-day / month / hour) added
   +0.010 on their own. Rates are smoothed towards the global rate (SMOOTH=60). Fitting these on the
   full training set *without* cross-fitting and training on them cost **-0.008** (exp 6): the model
   over-trusts in-sample group statistics. OOF cross-fitting fixes exactly that (exp 8, 10).

2. **Fine-grained time-of-day resolution in the rate keys.** This was the single largest lever:
   hourly keys → 30-min (+0.0069, exp 24) → 15-min (+0.0019, exp 26) → 10/5-min global and
   carrier×15-min (+0.0005, exp 28). Delay risk ramps steeply through the day, and coarse bins blur it.

3. **Dropping the raw high-cardinality categoricals.** Keeping Origin/Dest (282 levels) and carrier as
   native categoricals *hurt*: removing Origin+Dest gave +0.0039 (exp 19), removing carrier a further
   +0.0008 (exp 20, total 0.7330). The trees memorise 2005-specific per-airport levels that do not
   transfer to 2006; the smoothed rate/count features carry the same information in a drift-robust form.
   DayofMonth was worth dropping too (+0.0008, exp 22) — Month/DayOfWeek were not (exp 21, -0.0025).

4. **Ordinal calendar position** (`DOY = month*30.4 + day`, plus numeric month/DOW) gave +0.0020
   (exp 30): seasonality as one monotone feature instead of many categorical splits.

5. **Shallow-regularised trees + averaged ensemble.** depth 4 / lr 0.02 / 1500 trees / mcw 20 / λ 5 beats
   deeper models (depth 6: -0.0055, depth 8: -0.008): with the interaction structure already encoded in
   the features, extra depth only buys memorisation. Averaging 5→9 diverse members (depths 3–6, different
   colsample/subsample) added small, reliable variance reduction (0.7472 → 0.7479).

## What did not help

- **More capacity**: 400×depth 8 (0.7062 vs 0.7141 baseline) and depth 6 (-0.0055 on the final feature set).
- **In-sample (non-cross-fitted) target rates and kernel-smoothed minute-of-day curves**: 0.7065 and
  0.7373, both clearly worse than the OOF versions they were meant to improve on — pure leakage.
- **Time-of-day / arrival-time decomposition of `DepTime` as model features** (hour, minute, sin/cos) and
  distance-band rates: neutral (0.7140 vs 0.7142) — `DepTime` as hhmm is already monotone in time-of-day,
  and Duration is a monotone transform of Distance, so trees already had this information.
- Also neutral: 10-fold instead of 5-fold OOF, `max_bin=512`, season×time keys, route-carrier key, and
  removing the coarse hourly keys (-0.0107 — the hourly keys are *not* redundant with the fine bins).
- **Traffic-share (peakiness) and finer-bin share features**: only +0.0002–0.0004, reverted on the
  simplicity criterion.

## What I would try with more budget

The remaining headroom is in the *information* available, not the model: with these eight columns the
feature set is close to saturated (recent gains were ≤0.0003 each). The most promising direction is a
proper two-stage decomposition of the time-of-day dimension — a per-airport, per-month *curve* estimated
with a monotone/isotonic fit and cross-fitted per fold, rather than discrete bins, so the congestion ramp
is modelled at full resolution without the sparse-bin noise that made the kernel-curve attempt fail.
After that: weather/holiday proxies (none available here), and a bagged ensemble over several *rate
encodings* (different smoothing constants and fold seeds per member) instead of only over model
hyperparameters — a genuinely different view of the same data, which is where averaging still pays.
