# autoresearch XGBoost — airline dep_delayed_15min — FINAL

**Best Eval AUC: 0.7351** (baseline 0.7141, +0.0210). Best commit: `a3472bd` (16-member mega-bag,
prefix-selected down to its 8 base-feature members). validate.sh: `CONTRACT OK`, predict_proba AUC 0.7351.
40/40 experiments used; ~186 min wall; ~4.1k/18k CPU-s.

## The 5 changes that mattered most

1. **Calendar numerics** (+0.003): convert `c-<n>` strings to numbers — MonthNum, DayOfWeekNum, and a
   DayOfYear (cumulative-days table), letting trees split within the year instead of only at month
   boundaries. Dropping the now-redundant raw `Month`/`DayofMonth` categoricals added a further
   +0.0005–0.001 by not diluting splits.
2. **Origin x DepHour interaction categorical** (+0.0021): `Origin_h<HH>` as a native categorical
   (levels from train only). Captures airport-specific time-of-day delay patterns; the single biggest
   feature-level win. Only OrigHour helped — Dest_x_hour, Carrier_x_hour, Route, Origin x DOW all
   regressed (rare levels + 2005→2006 carrier/airport churn).
3. **Shallow, regularized boosters** (+0.002–0.003): depth 3–5, 300–600 rounds, lr 0.02–0.05,
   reg_lambda 5, alpha 3–5, subsample 0.8. The 2005→2006 time shift punishes capacity: depth 8 and
   1000+ round configs always lost. Every capacity re-probe after a feature change found a new
   optimum, so features and hyperparameters had to be re-tuned together.
4. **Multi-config bagging** (+0.001–0.002): averaging 5–8 XGBoost models with diversified
   depth/rounds/lr/alpha (plus one deep 200x d8 and one shallow 1200x d3 member) beat every single
   model and every seed-only bag. Simple mean of probabilities was as good as rank-mean.
5. **MinOfDay** (+0.001): continuous minute-of-day (hh*60+mm) alongside DepHour, letting trees use
   within-hour granularity of scheduled departure time (DepTime has 0.41 importance in every run).

## Things that did not help (all reverted)

- **Target encoding** of Carrier/Origin/Dest/Route (−0.009 with 30 trees, −0.014 with the tuned
  config): the smoothed priors from 2005 shift by 2006 and the shallow trees latch onto them.
- **More raw capacity / early stopping on random or chronological splits**: 500x d6 (−0.007), and
  chronological early stopping on the last 15% of 2005 collapsed to 0.7169 (it stopped at ~50 trees —
  late-2005 is a bad proxy for 2006). Random-split ES also hurt.
- **One-hot encoding** of the categoricals: crashes on eval (airports present in 2006 but absent in
  2005 produce missing dummies); native categoricals handle unseen levels gracefully. Also no gain
  where computable.
- Also tried without effect: seed-only ensembles (hist is near-deterministic here), recency-weighted
  training, column-bagging, log-Distance, hour sin/cos, rush-hour flags, hour bins, DestHour /
  CarrierHour / Route features, max_bin/gamma/min_child_weight tweaks, expanding the bag to 16–24
  members.

## What I would try with more budget

The eval signal is clearly feature-limited, not capacity-limited, and interactions only pay off in
the Origin x hour direction — which suggests the underlying signal is congestion/queueing at the
departure airport in the time-of-day band. I would engineer that directly: per-airport scheduled
departure counts per hour (a proxy for hub congestion and queue load) computed from train, and
"minutes since the previous scheduled departure at the same airport" (both derivable from the
2005 slice; both survive the year shift because airports are physical). Next I would pursue honest
time-validation instead of eval-only selection: block-CV across 2005 months (train Jan–Oct, validate
Nov–Dec) to get an unbiased estimate of each change, since several of my eval gains near the end
(0.7349→0.7351) are within seed noise and may not transfer to the hidden holdout. Finally, more
bagging diversity that actually decorrelates — feature-set rotations around the winning feature
group rather than seed families, and per-member subsample rates spanning 0.6–0.9, tuned with the
blocked validation.

## Reproduction

`python train.py` reads data/train.csv + data/eval.csv, trains 8 XGBoost members on the engineered
features (all inside `prepare(df)`, fit on train only), averages their probabilities, prints
`Eval AUC: 0.7351`. `predict_proba(df)` applies the identical path to raw holdout rows.
