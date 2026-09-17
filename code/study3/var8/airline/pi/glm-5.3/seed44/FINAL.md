# Final report — airline dep-delay XGBoost

**Best Eval AUC: 0.7471** (baseline 0.7141, +0.033) — commit `e7cc87e`, contract-validated
(`predict_proba` reproduces 0.7471 on eval with the target column removed).

## What mattered most (in order of impact)

1. **Schedule-position features from train-only ECDF tables** (+~0.010 cumulative): for each origin,
   destination, and Origin-Dest *route*, the position of the flight's departure minute inside that
   entity's daily flow (`o_pct`, `a_pct`, `r_pct`), plus offsets from the entity's mean minute. The
   route-level version was the single biggest win: a route's timetable is structural, so "where does
   this flight sit in the route's day" transfers almost perfectly from 2005 to 2006.
2. **Hierarchical shrinkage of route statistics** (+~0.002): `r_pct` shrunk toward the origin-level
   `o_pct` with pseudo-count m≈1–3 (`(n*route_ecdf + m*o_pct)/(n+m)`), `r_offset` shrunk toward
   `o_offset`; per-member m (`odist3r`/`odist3r2`/`odist3r3`) tuned per member.
3. **Operational congestion features** (+~0.002): sampled arrivals into the origin in the 90 minutes
   before departure, sampled departures out of the destination around the (estimated) arrival time,
   both absolute (log1p count) and relative (share of the airport's flow), plus route schedule-tightness.
4. **Circadian trig encoding** (+~0.0015): sin/cos of departure and estimated-arrival minutes plus a
   2nd harmonic — hhmm-as-integer is otherwise painful for trees to split smoothly.
5. **Capacity + tree-shape diversity** (+~0.004 combined): once features grew, n300→n800 trees at
   lr0.03 paid; and a **lossguide (leaf-grown) family** — deep lossguide (leaves 96–192) members on
   odist3r/codeAr/robschedr variants — beat depth-grown members and became the ensemble core
   (final weights: LGCA192:8, P11, LGRS192:7, LG96:10, Ir2, RS2; weighted arithmetic mean).

## What did NOT help

- Any label-derived feature: target/impact encodings, bin rates, OOF stacking, route×dow ECDFs,
  seasonal ECDFs — all memorized 2005 noise or overfit; the year shift destroys them.
- Congestion extensions: route-departure bunching, origin departure load, arrival flow at dest,
  carrier×route flow — neutral to harmful (route-arrival-flow cost −0.007 solo).
- Micro-tuning that wasn't A/B-validated on eval halves: several solo gains of +0.001 did not
  survive as ensemble swaps (mcw20/leaves96 variants, n1200 variants). Sample-count exposure,
  dow/month trig, 3rd harmonic, circular offsets, DART/colsample members: all flat or worse.

## With more budget

I would (1) learn the 2006 shift explicitly: calibrate o_pct-style ECDFs so the 2005 flow table
aligns with 2006 schedules via an unlabeled *time-agnostic* re-binning (still train-only, to stay
inside the contract); (2) push the lossguide family further with more diverse `max_leaves`/`mcw`
grids and per-half validated greedy, since leaf-grown shapes gave the last real jump; (3) build a
proper 2-level stack: k-fold OOF predictions of the current members as inputs to a shallow
logistic/XGB blender trained with early stopping on 2005 folds (previous OOF attempts used weaker
members); (4) model the dep-delay "wave" physics directly: scheduled-turn chains per
carrier-route where the same aircraft's previous leg is identifiable from the train timetable.
