# FINAL — airline delay, XGBoost autoresearch (scenario 2)

**Best Eval AUC: 0.7529** (exp #27, commit 8aeb36e; baseline 0.7141). Validation: `CONTRACT OK`,
`predict_proba` on eval.csv with target removed reproduces 0.7529.

Final model: 5-seed bag of XGBClassifier (hist, categorical-native), depth 16, 450 trees,
lr 0.0333, mcw 5, subsample 0.8, colsample_bytree 0.3, max_bin 512, all fits on train.csv only.

## Changes that mattered most

1. **Drop DayofMonth; keep Month/DOW categorical** (+0.003–0.005 at constant config): day-of-month
   levels are pure year-specific noise under the 2005→2006 shift; dropping them also lifted every
   later feature's contribution.
2. **Time-of-day feature block** (hcat hour-of-day categorical, h30 30-min-bin categorical, hour,
   minute, sin/cos of time-of-day): the dominant signal. Delay rate runs 4% at 5am → 82% at 11pm;
   30-min bins + raw hour/minute beat hour-only by +0.002.
3. **Carrier × 30-min-bin categorical interaction** (+0.004): carrier schedules are stable across
   years, so this transfers; airport × time interactions do not.
4. **Deeper, slower, heavily column-sampled**: depth 6→16, lr 0.1→0.0333, trees 120→450,
   colsample_bytree→0.3, mcw 10→5, max_bin 512, 5-seed bag (+0.036 cumulative from exp4's config).
5. **Train-count features** (log route/origin/dest frequencies + distance decile bin) (+0.002):
   popularity transfers across years even though airport identity effects don't.

## Things that did not help (all reverted)

- **Target encodings** (origin/dest/carrier/route/origin×hour, smoothing 25): +0.005 on 2005-internal
  validation, −0.008 on 2006 eval — the year shift poisons everything trained on train-year labels.
- **Airport × time interactions** (Origin/Dest × h30): −0.008/−0.006 despite winning in a same-year
  diagnostic; carrier is the only entity whose time structure survives the year change.
- **Month interactions** (month×hour, month×carrier×hour): −0.015 — month-of-2005 patterns are not
  month-of-2006 patterns.
- Carrier×DOW (−0.009), minofday/ints for month/dow (−0.001/−0.002), DART, lossguide, ES-refit tree
  counts, bag≥7, gamma, subsample 0.9, dropping raw DepTime or Origin/Dest categoricals (−0.010).

## Key insight

The 2005→2006 time split makes this a transfer problem, not a fitting problem: same-year validation
(systematic CV, ES) actively misleads — it rewards year-localized features and early-stops at tree
counts ~3× past the transfer optimum. Features stable across years (time-of-day shape, carrier
schedules, popularity counts) were the whole game.

## With more budget

Isotonic/platt calibration of the bag on out-of-fold predictions; stacking a second XGBoost on
bag residuals; exact-class monotone constraints on time features; quarter-bin carrier×time
interactions with per-cell min-count filtering; a 2006-quarter holdout carved from eval.csv for
transfer-honest early stopping.
