# Final Report — szilard/xgboost-autoresearch scenario 2 (airline)

## Best result

**Eval AUC = 0.7415** (experiment #32, commit `491cfba`, validated via `./validate.sh` → CONTRACT OK).
Baseline (30 trees, depth 6) was 0.7141, so the loop gained **+0.0274 AUC**.

## Final model

Bagged ensemble of 3 XGBoost `hist` members, averaged in probability space via pre-built
`xgb.DMatrix` objects (`m.get_booster().predict(dm)` — avoids the sklearn wrapper's per-call
pandas→DMatrix conversion, which alone cost ~7 s/member on 100k rows):

1. d22, lr 0.015, n=320, mcw 2, colsample 0.8, bin 256
2. d22, lr 0.02, n=280, mcw 2, colsample 0.75, bin 256
3. d22, lr 0.02, n=260, **mcw 1**, colsample 0.8, bin 256

All members: `subsample=1.0`, `reg_lambda=1.0`, full 100k-row 2005 training data, native categoricals
(UniqueCarrier, Origin, Dest; levels fit on train only). Features: numeric Month/DayofMonth/DayOfWeek
+ sin/cos encodings (month, dow), raw DepTime, Dep_hour, Dep_min_of_day + sin/cos, Distance.
A dynamic time guard (fit estimate + per-member predict cost + overhead ≤ 112 s) is what makes
3×~26 s members reliably fit inside the 120 s experiment cap.

## The 3–5 changes that mattered most

1. **Very deep trees** (max_depth 6 → 22): the single biggest lever. Monotone gains through d16,
   d18, d20 to d22 (+0.0013 alone from d16→d20 ensembles, members 0.7347 → 0.7377). d24 no longer
   helped. Deep trees + low lr + mcw 1–2 dominate this dataset.
2. **Bagging decorrelated members** (seed/colsample/lr/n/mcw variation, probability-averaged):
   consistently +0.003 over the best single member (0.7388 member → 0.7415 ensemble). Ensembles
   throughout the campaign: 7-seed bag +0.003, hetero +0.004, deep-region +0.004 over their members.
3. **Full-data members**: `subsample=1.0` beat 0.8–0.9 members (exp 25 kept; exp 35 confirmed:
   0.7393 vs 0.7415) — row subsampling only hurt member strength without adding useful diversity.
4. **Low learning rate with moderate round counts**: lr 0.03 n340 (0.733) → lr 0.02 n280 (0.7388).
   The classic lr↓/n↑ trade was worth ~+0.002 at every depth tried.
5. **Cyclical + decomposed time features** (Dep_min_of_day, sin/cos at 1440/12/7 periods) from the
   raw `c-N` strings: +0.004 at the start (0.7141 → 0.7185) and everything later built on it.
   Dep_min_of_day is by far the dominant signal (evening departures delay more).

## Things that did not help (all measured, then reverted)

1. **Fine-grained categorical interactions / route identity**: explicit Origin–Dest route category
   (0.7059!), OOF target encodings (0.7075), carrier×hour and dow×hour interaction cats (members
   crashed to ~0.715): 2005-specific granularity does not transfer to 2006. Plain Origin and Dest as
   categoricals are essential though — dropping them costs −0.031 (0.7049).
2. **Holiday-distance / day-of-year features**: −0.008 (0.7331). Thin per-day data at depth 22 turns
   them into spurious splits.
3. **Snapshot ensembling** (one lr 0.01/n560 trajectory, 4 iteration-range snapshots averaged):
   0.7402 — snapshots are too correlated to beat 3 independently-seeded fits.
   Also flat/negative: `colsample_bynode` (0.7308), mid-depth d14–15 filler members (0.7367 ≈ equal),
   max_bin 64 (−0.003 per member), gamma 0.1–0.3 (neutral), time-ordered internal validation
   (late-2005 split has a seasonal shift; its early stopping at ~110 rounds is misleading).
4. (timeout lessons: member fits + per-member predict must be budgeted together — a naive 4-member
   config reliably blew the 120 s cap; the dynamic guard fixed it.)

## What I would try with more budget

The plateau at ~0.7415 is the 120 s/experiment cap as much as the model: 3 members × ~26 s fit is
all that fits alongside ~17 s of predicting and ~10 s overhead. With a larger per-experiment budget
I would (a) fit 6–10 d22 members (bagging gains were still visible going 3→4 whenever the 4th
actually fit, e.g. exps 25/27 at 0.7366/0.7367 vs 3-member equivalents), (b) probe d26–d30 with
n 200–240 at lr 0.02–0.025 since the depth ladder only flattened (never turned down) between d20 and
d24, and (c) tune reg_lambda/alpha jointly with depth, which I only left at the default. On the
feature side I would test *removing* the redundant raw DepTime/Dep_hour columns (exp 16 suggested
neutrality, but it timed out before a clean read), and revisit carrier-level hour-shift features at
much coarser granularity (3 bins), since every failure above came from over-fine category levels.
None of these promise > +0.002; the honest read is that ~0.742 is close to what this 100k-row,
7-column slice of the problem supports for a 2005→2006 transfer.

## Experiment ledger

37 of 40 experiment slots used (14 kept, 22 reverted/timeouts). Full history with per-member
diagnostics in `experiments.tsv` and `run.log`; notable milestones: #1 baseline 0.7141, #2 features
0.7185, #9 seed bag 0.7240, #13 deep region 0.7343, #18 lr sweep 0.7360, #25 full-data members
0.7366, #29 d18 0.7380, #30 d20 0.7406, #31 d22 0.7411, #32 d22 refinement **0.7415** (final).
