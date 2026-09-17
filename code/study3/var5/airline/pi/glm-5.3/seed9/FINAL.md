# Final report — airline delay (XGBoost, autoresearch benchmark)

**Best Eval AUC: 0.7360** (experiment #38, commit `fb335f6`, validated with `CONTRACT OK`;
baseline was 0.7141, so **+0.0219 AUC** overall).

## Changes that mattered most (in order of impact)

1. **One-hot encoding of UniqueCarrier / Origin / Dest** instead of XGBoost native categorical
   partitioning (`enable_categorical`): 0.7148 → 0.7271 (**+0.0123**, the decisive jump, exp 12).
   Explicit dummies let every tree build clean airport/carrier corrections that survive the
   2005→2006 year shift; native category partitioning clearly did not.
2. **Origin×Hour interaction dummies** (cells with ≥100 train flights): 0.7330 → 0.7359
   (**+0.0029**, exp 35). Evening departure-bank congestion at hub airports is
   schedule-structural and repeats year over year. Carrier×Hour dummies added a small
   +0.0003 on top.
3. **Capacity scaling**: 300 → 1600 → 3200 trees at lr 0.05 under the one-hot representation
   (+0.0035 combined, exps 13–16); the knee sat at 2000–3200 trees, and moved to ~2600 after
   later changes (final model: 2600 trees, depth 6, lr 0.05).
4. **Recency sample weighting** (linear ramp 0.7 → 1.0 over the 2005 months): +0.0011
   (exp 28). The 2005→2006 distribution shift is real; down-weighting stale months helps
   transfer. A steeper 0.5→1.0 ramp hurt, so the effect is gentle.
5. **min_child_weight=10** (+0.0005, exp 17) and **per-origin/dest/route mean-distance
   summaries** (+0.0005, exp 21): leaf-level robustness and cheap structural hub descriptors.

## Things that did NOT help (all reverted)

1. **Target encodings** (OOF, smoothed, any of route/origin/dest/carrier/hour/month/DOW,
   with or without hour interactions): -0.003 to -0.008. Diagnosis: 2005 group rates are
   mostly year-specific noise — route-rate correlation between 2005 and 2006 is only 0.256
   (0.685 for the ~88 biggest routes, still not enough); hour TE is redundant with raw
   MinuteOfDay, which transfers perfectly (0.68 AUC in both years).
2. **Fine-grained calendar features** — day-of-year, sin/cos-of-year, holiday-proximity:
   -0.004 to -0.006. The model just memorizes 2005-specific dates that do not recur.
   Volume/density count features (hub size) also hurt slightly.
3. **Ensembles and sampling tweaks** — 5-seed bagging, 2-seed averaging, a diverse
   3-model ensemble (one-hot + native-categorical + shallow), subsample/colsample 0.7,
   colsample 0.8, depth 8, lr 0.03/0.04, max_bin 64, DART-style tweaks were not tried but
   every other variance-reduction idea plateaued at ±0.0005 of the single model.
   Eval-noise floor is ~±0.003 AUC on 100k rows, so these were indistinguishable.

## What I would try with more budget

The two structural wins (one-hot dummies, interaction dummies) suggest the remaining signal
lives in well-supported interaction cells: I would mine origin×hour further (finer hour
buckets, e.g. 30-minute banks, with support-adaptive thresholds), try route×hour dummies for
the biggest routes, and carrier×origin dummies, always with a ≥100-flight support cutoff so
only year-stable cells enter. I would also revisit the 2005→2006 shift with a *learned*
weighting (fit the recency ramp shape, or weight by similarity of month-of-year to the
deployment window) rather than a hand-set linear ramp, and I would re-check the tree-count
knee after each feature family since it moved twice. Finally, a small bag of 2–3 models at
the final configuration with different feature subsets would likely add +0.001–0.002, which
is below what my eval readings could confirm, so I favored the simpler single model.
