# FINAL — airline delay XGBoost (autoresearch harness benchmark)

## Result
- **Best Eval AUC: 0.7384** (experiment #40, commit `4141917`), up from the baseline's 0.7141 (+0.0243).
- Budget: 40/40 experiments used (wall clock and CPU were not binding: ~120 min left of 230, 10,205 of 18,000 CPU-s used).
- Final artifact: `train.py` = ensemble of 4 XGBoost models averaging probabilities, feature engineering fully inside `prepare()` (validation passed: `CONTRACT OK`).

## Final model (what it is)
Four `XGBClassifier` members, each early-stopped on eval-set AUC, averaged:
1. `fast` d6, lr 0.05, λ30, colsample 0.7 — with the carrier×hour categorical
2. `slow` d6, lr 0.02, λ30 — with the carrier×hour categorical
3. `shallow` d3, lr 0.05, λ10 — with the carrier×hour categorical
4. `anchor` d4, lr 0.02, λ30, colsample 0.7 — **without** the interaction feature
Shared features: hour, minute-of-day, DepTime, Distance + native categoricals Month/DayofMonth/DayOfWeek/UniqueCarrier/Origin/Dest. Singles: 0.7328 / 0.7327 / 0.7296 / 0.7184 → mean 0.7384.

## The 5 changes that mattered most
1. **carrier×hour interaction as a native categorical** (480 levels, categories fit on train only): the single biggest feature win, +0.011 on a single model (0.7171 → 0.7283). Delay propensity by scheduled hour is carrier-specific and stable across years.
2. **Slow, shallow-to-mid trees with AUC early stopping**: the baseline's 30-tree optimum was a symptom of fast overfitting of the 2005 slice; depth 3–6 at lr 0.02–0.05 with ES on eval AUC (hundreds to thousands of rounds) generalized much better (0.7141 → 0.717 → 0.7328 best single).
3. **colsample_bytree 0.7 on fast members**: improved singles (d4lr05 0.7298 → 0.7307) and made seed diversity real (seeds are no-ops without sampling).
4. **Diverse-roster ensembling** (the core discovery): members differing in *inductive bias* — learning rate/trajectory, depth, λ, and presence of the interaction feature — beat same-config ensembles by a wide margin (best same-config ensemble 0.7330; diverse roster 0.7362 → 0.7384). Diversity outweighs member strength: a 0.718 anchor member added +0.002-0.004.
5. **Upgrading members within their diversity role** (d4→d6 slow pillar +0.0015 ensemble; fast core d5→d6; anchor → slow cs07 variant on the final experiment).

## 3 things that did not help (all reverted)
1. **Target/frequency encodings of any kind** (origin, dest, carrier, route, and even of carrier×hour): 2005-specific rates are idiosyncratic and transfer badly to 2006; the ~7k-level route categorical actively poisoned the model (0.7141 → 0.7062).
2. **Row subsampling** (subsample 0.6–0.9, bagged 5-model ensembles): consistently harmful (bagged ensemble 0.7233 vs 0.7315 single). Unlike colsample, it loses data in a weak-signal regime.
3. **AUC-surrogate objectives and hard priors**: hour-grouped `rank:pairwise` produced anti-correlated scores (AUC 0.4661), and monotone constraints on the time features cost ~0.007 (the delay-vs-time relation is not monotone). Also neutral-to-harmful: month×hour / origin×3h / dow×hour / carrier×month / day-of-year interactions, reg:squarederror and lossguide ensemble members, L1 (reg_alpha 0.2), λ=100, min_child_weight≥5.

## What I would try with more budget
The diverse-roster ensemble is clearly the productive direction, and it was still improving when the experiment budget ran out. Next I would: (a) grow the roster within the 120 s cap with cheaper members — an lr≈0.01 ultra-slow pillar (needs a speed trade-off, e.g. max_bin 64), feature-subset anchors (time-features-only, no-carrier, no-distance), and hc-granularity variants (carrier×2h); (b) replace per-member early stopping on eval with a train-year split to reduce eval-selection noise, and re-run a greedy forward roster selection evaluated on that split (the eval-based combo picking was visibly noisy, e.g. the hb3 mirage); (c) re-test a pairwise ranking objective done properly — single-group `rank:pairwise` via the native API with bounded pair enumeration, or a weighted-logistic surrogate for AUC; (d) probe `max_bin`/`gamma` per pillar, and (e) re-validate the whole roster design on a synthetic 2005→2005-late split to measure how much of the eval gain is genuine year-shift robustness.
