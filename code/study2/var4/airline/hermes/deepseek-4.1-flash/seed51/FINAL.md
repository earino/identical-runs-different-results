# Final report — airline departure-delay classifier

**Best Eval AUC: 0.7600** (experiment #38, commit `56dd734`; baseline was 0.7141).
40/40 experiments used, ~6,200 of 18,000 CPU-seconds, ~40 of 230 minutes wall clock.
`./validate.sh` prints `CONTRACT OK`; the contract path (`predict_proba` on `data/eval.csv` with the target
column removed) reproduces 0.7600.

## What `train.py` does

Every feature is a **target-free structural property of the flight network**, computed from the raw columns and
fitted only on 2005 traffic statistics. All of it lives inside `prepare(df)`, so the hidden holdout gets exactly
the same transformation:

- ordinal/cyclical decode of the `c-<n>` seasonal columns (month, day-of-month, day-of-week), `hhmm` departure
  decomposition (`dep_frac`, sin/cos, next-day flag), block time from distance.
- traffic-volume counts per (origin, dest, carrier, route, carrier×origin) and per time slot.
- **slot-level density ladders**: for each key, counts over 5/15/30-minute slots and windows of ±15/±30/±60
  minutes, plus arrival-side density at the destination (estimated arrival slot ≈ departure + distance/450 mph).
- **asymmetric (back vs forward) windows** at 15/30/60/120 minutes and whole-day, for origin, carrier-bank,
  route and arrival-side traffic, plus schedule-gap (headway) features.
- 5-member diverse `XGBClassifier` ensemble (depth 5–8, lr 0.02–0.04, colsample 0.45–0.7) averaged in
  `predict_proba`.

## The 5 changes that mattered most

1. **Asymmetric back/forward traffic windows** ("am I at the head or the tail of a departure bank?").
   §27 +0.0051, §28 +0.0056, §35 +0.0038, §37 +0.0011 — the single most valuable idea, ~0.012 total.
2. **Fine-grained slot congestion instead of hourly counts** (§17 +0.0109, §18 +0.0016, §21 +0.0006): half-hour
   and 5-minute buckets with rolling density windows, plus the destination arrival-side proxy.
3. **Carrier-bank and inbound-turnaround density** (§19 +0.0025): traffic of the *same carrier* at the origin
   window, and flights estimated to be landing at my origin around my pushback.
4. **Ordinal + cyclical time encodings** (§15 +0.0020): continuous `dep_frac`, month/day-of-week sin-cos and
   day-of-year alongside the categorical versions.
5. **Model capacity plus a small ensemble** (§10 +0.0024, §14 +0.0017, §32/§34 +0.0008): lower learning rate
   with more trees, then averaging 5 diverse boosters.

## What did not help

1. **Target-derived encodings** (smoothed target/frequency encoding of route, carrier, airport and hour) —
   0.7076 vs 0.7141 at the time. 2005 delay *rates* do not transfer to 2006, and the in-sample fits made the
   trees over-rely on them. All kept features are deliberately target-free.
2. **Native high-cardinality categoricals** (`route`, `carrier_origin`, `origin_hour` as `category` dtype with
   `enable_categorical`) — −0.0055. Partition splits on thousands of levels memorise the training year.
3. **Restricting the model with monotone constraints** on time-of-day and congestion (0.7404 vs 0.7426), and
   **cumulative day-pressure / burstiness-ratio / exponential-decay features** (§20, §25, §31 — all negative).
   Extra correlated congestion aggregates add noise rather than signal.

## If I had more budget

The clearest open direction is to make the delay-propagation signal explicit rather than aggregate: the current
model sees how much traffic surrounds a flight, but not *which aircraft* it inherits a delay from. With more
budget I would attempt a target-free aircraft-rotation proxy — chaining consecutive flights by (carrier,
origin/destination, time-slot compatibility) into a schedule graph and feeding the chain position, the estimated
slack before the next leg, and the number of same-carrier inbound legs landing at the origin in the preceding
window. That is the mechanism the asymmetric windows are only crudely approximating, and it could plausibly add
another 0.005–0.01. Secondary directions: a properly out-of-fold target encoding of route/carrier delay
propensity (fitted per-fold so the training rows never see their own label) to test whether the §3 failure was
leakage or genuine year shift, a 2006-style validation split built by holding out late-2005 months to tune
capacity without touching `eval.csv`, and calibrating the tree count with a real early-stopping pass instead of
the hand-set ladder used here.
