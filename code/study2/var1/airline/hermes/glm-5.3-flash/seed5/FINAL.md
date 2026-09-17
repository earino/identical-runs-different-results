# Final Report — airline (autoresearch XGBoost, scenario 2)

**Best Eval AUC: 0.7453** (baseline 0.7141, +0.0312). 15 counted experiments, all within the 120 s / run
limit; the CPU-seconds budget (18000) was the binding constraint at the end, not the 40-experiment cap.

## Final architecture (HEAD = a875791)

- Features: DepTime (clipped at 2359), Distance, dephour (DepTime//100), qslot (DepTime//5), sin/cos of
  minute-of-day; shrunk target encodings (m=100 toward the 0.5 base rate) of Month, DayofMonth, DayOfWeek,
  UniqueCarrier, Origin, Dest; joint TEs carrier x hourbin (m=300), origin x hourbin (m=300), carrier x
  dephour (m=500); route frequency (train count of Origin_Dest); joint-TE cell counts (support size per
  encoded cell). All encoders fit on train.csv only and reused verbatim inside `predict_proba`.
- Model: XGBClassifier, lossguide growth, 128 leaves, lr 0.03, 3000 rounds, subsample 0.95,
  colsample_bytree 0.75, max_bin 512, min_child_weight 1; ensemble of 3 seeds (42, 0, 2), probability-averaged.

## Changes that mattered most

1. **Shrunk target encodings instead of raw categoricals** (+~0.015 over the baseline path): compact
   numeric category representations let small, strongly regularized models generalize across the
   2005 -> 2006 shift. Out-of-fold encoding at train time measured WORSE (0.639-0.715); the deployed
   full-map encoding is self-consistent between training and prediction.
2. **Joint target encodings** (carrier x hourbin, origin x hourbin, carrier x dephour): +~0.005.
   Interaction cells with heavier shrinkage (m=300-500).
3. **Leaf-wise growth + long low-LR schedule** (lossguide, 128 leaves, lr 0.03, n~3000): +~0.004 over
   depth-6..10 hist trees, which all plateaued near 0.715-0.726.
4. **Multi-seed ensemble** (3 seeds, +0.0016): averages away fit-order noise; 4 seeds timed out at 120 s.
5. **Time-of-day numerics** (qslot = DepTime//5, sin/cos angle) and **TE cell counts**: +~0.001-0.002 each;
   the cell counts let the model separate trustworthy encodings from backfilled ones.

## Things that did NOT help

- Out-of-fold target encoding at train time (0.639-0.715) and OOF+full hybrid columns (-0.005).
- Route target encoding (-0.013 despite route frequency helping); origin/dest x dow/month/carrier joint
  TEs, month x hourbin, distance-bin TE, dow/dom ordinals, single-cat counts, log1p/capped counts.
- Regularization micro-tuning: min_child_weight 2-20, gamma 0.5-2, reg_lambda 5, colsample_bylevel/bytree
  variations, depth sweeps, lr 0.1-0.3 with few rounds; blending with a raw-categorical model; scipy-free
  rank averaging (equal to mean); TE m=60-400 sweeps (flat).

## With more budget

- A proper year-shift robustness study: fit 2005 vs 2006 rate ratios per carrier/route to re-anchor TEs
  (the residual headroom is almost certainly the year shift).
- Larger seed/member ensembles with staggered hyperparameters under the time limit (n=2400/lr=0.035
  members), plus quantile-bin DepTime smoothing of the TE joints.
- Native-XGBoost pipeline (xgb.train on QuantileDMatrix) to cut data-prep overhead and squeeze more
  seeds under the 120 s cap.
