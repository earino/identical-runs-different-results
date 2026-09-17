# FINAL — airline dep_delayed_15min (XGBoost, 40/40 experiments used)

**Best Eval AUC: 0.7356** (experiment #40, commit `b32fce4`), up from the 0.7141 baseline.
Baseline → best is +0.0215 AUC. Train is 2005, eval/holdout are 2006; the model's own
out-of-fold AUC on 2005 is ~0.75, so the ~0.735 ceiling on 2006 is set by the year shift,
not by fitting capacity.

## Changes that mattered most

1. **L1 sparsity (`reg_alpha`) — the single biggest lever (+0.005).** With `reg_alpha=0` the
   model peaks at depth 4 / ~300 trees and degrades with anything more; sweeping alpha showed a
   clean curve (1→0.7201, 2→0.7216, 5→0.7219, 10→0.7230, 20→0.7212, 50→0.7198). alpha=10 also
   *raises* the useful tree budget (1200 trees became better than 300). Sparsity suppresses the
   2005-specific splits that do not transfer.
2. **Smoothed out-of-fold target encodings fit on train only.** `Origin×hour` (k=30) was the one
   interaction key that generalized (+0.002 over raw features and +0.0017 over plain
   origin/dest/carrier encodings), and `Origin×5-minute-bin` (k=100) added a further +0.0017
   (airport schedule-bank congestion). Out-of-fold values are used for the training rows so
   `predict_proba` stays honest on unseen data.
3. **Departure-time decomposition** (`dep_hour`, `dep_minute`, cyclic tod/month/dow). These were
   *neutral* before regularization and only paid off once alpha>0 (+0.0013). `dep_minute` alone
   was worth ~+0.002 — scheduled minute-of-hour encodes departure banks.
4. **Shallow trees with a mid learning rate** (depth 4–7, lr 0.02–0.03). Depth 6 was consistently
   better than a single deep model: first experiments showed eval AUC decaying monotonically with
   tree count at depth 6 (0.7141 at 30 trees → 0.7070 at 600), i.e. the raw baseline was already
   overfitting.
5. **Small ensemble of diverse regularized members (final +0.0005).** Six members (depth 6/7,
   alpha 10/12, n 800–1500, two of them with subsample/colsample 0.9) beat every single member
   (best member 0.7353 vs ensemble 0.7356) and beat a homogeneous 8/20-member bag.

## What did not help

- **More capacity in any form**: deeper trees without L1, thousands of trees, `max_bin=64`,
  `booster="dart"` (0.7159 — clearly worse), `grow_policy="lossguide"`, `max_delta_step`.
- **Most extra encoding keys**: route target encoding (and the route categorical+frequency
  version, which cost a full 0.012), `carrier×hour` (0.7151), `Origin×month`, `Origin×dow`,
  `Origin×weekend×hour`, `Carrier×origin`, global `Hour` — all neutral or harmful.
- **Ensembling/stacking as a generic answer**: bagging 20 homogeneous members (0.7194 < 0.7201),
  level-2 stacking over 4 CV-fold bases (0.7197 < 0.7201), and L1-member ensembles at
  `n_estimators=1200` (0.7244 vs 0.7245 single) were all washes or worse; only diversity across
  *different* configs helped, and only marginally.
- Unused/dead features: `log_distance` and `is_weekend` were dropped after ablation showed no
  loss.

## With more budget I would try

Re-frame the problem as distribution shift rather than fitting. Concretely: (a) an
adversarial/importance-weighted fit between 2005 and 2006 features to reweight training rows
toward the 2006 region of feature space; (b) monotone or interaction constraints built from the
domain (e.g. force the airport-hour encoding to interact with hour-of-day), which shrink the
hypothesis space in a way plain L1 cannot; (c) sweeping the target-encoding smoothing *jointly*
with the keys (the k=30/k=100 values were tuned one key at a time); (d) per-segment calibration by
carrier or airport, since the year shift likely hits some airports harder than others; and
(e) ensembling at the *prediction* level with weights chosen on a held-out slice of 2005 with a
deliberate 2005/2006 gap, instead of on eval.csv, to keep the anti-overfitting discipline that
drove most of the gain here. The remaining headroom looks small (every honest probe landed in
0.7345–0.7356), so the expected marginal value of further tuning is low relative to these
shift-aware ideas.
