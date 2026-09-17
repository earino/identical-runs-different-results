# Final report — airline dep_delayed_15min (XGBoost)

**Best Eval AUC: 0.7244** (baseline: 0.7141), 40/40 experiments used, ~5100/18000 CPU-s.

## What mattered most

1. **Feature-diverse bagged ensemble of small XGBoosts** (exp 10–16): 50 members with randomized
   configs (25–40 trees, depth 4–8, lr 0.1–0.15, subsample/colsample 0.6–1.0) and 3 feature-set
   variants (raw / +time numerics / +target encodings). Uniform config+feature diversity was worth
   ~+0.004 over the single-model baseline. Homogeneous bags and dart members were worse.
2. **Day-of-year harmonics** (exp 24): sin/cos of (Month,DayofMonth)→day-of-year gave the largest
   single jump (+0.0019) — calendar continuity that separate Month/Day categoricals can't express.
3. **Holiday-window flags** (exp 26): Nov 21–30, Dec 19–31 + Jan 1–2, Jul 1–5 (+0.0002).
4. **Greedy ensemble selection on 3-fold OOF predictions** (exp 20, 29): Caruana-style selection
   with replacement over a 70-candidate pool (+0.0006 over uniform mixing).
5. **3-seed averaging of each selected member** (exp 31, +0.0008) and **dropping DayofMonth from
   all variants** once DOY harmonics existed (exp 36–37, +0.0011): removing redundant noisy
   features helped as much as adding good ones.

## What did not help

1. **Bigger/longer single models**: 400–600 trees overfit hard (0.7005–0.7071 vs 0.7141); even
   early-stopping (119 trees) transferred worse than 30 trees. The 2005→2006 shift punishes capacity.
2. **OOF smoothed target encoding as a solo feature set** (exp 8, 0.7124) and carrier×hour TE
   (exp 22): neutral-to-negative; TE members still earned a slot via ensemble diversity.
3. **Stacking** (exp 21): XGB meta-model on logit OOF member preds lost to plain averaging
   (OOF-vs-full-fit distribution shift). RF-style members, 100-member ensembles, 5-seed averaging,
   hub-size counts, extra holiday windows: all ties or worse.

## With more budget

I would enlarge the candidate pool (100+ configs, richer variant space incl. DepTime-binned and
route-frequency features) with 5-fold OOF for more reliable selection, average two independent
greedy paths (the final union-of-two-paths experiment landed 0.7242 vs 0.7244, suggesting
selection variance is a real but small cost), and explore weighted blending of the greedy bag with
the uniform 50-member bag. Given the year-shift regime, robustness moves (seed averaging, feature
ablation) consistently beat capacity moves.
