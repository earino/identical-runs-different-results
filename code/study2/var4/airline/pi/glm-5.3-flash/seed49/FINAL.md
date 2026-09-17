# Final Report — airline delay (XGBoost binary classifier)

**Best Eval AUC: 0.7282** (baseline: 0.7141, +0.0141 over 40 experiments)

Final model: a 22-member heterogeneous XGBoost ensemble averaging probabilities over four feature
views (pruned-raw+hubness, TE+hubness, TE+hubness+DepTime+Distance, all), trained on random 80% row
subsets with temporal drift weights, OOF target encoding fit on train only, and `predict_proba(df)`
reproducing the full pipeline from a raw DataFrame (validate.py: CONTRACT OK, 0.7282).

## Changes that mattered most

1. **Heterogeneous ensembling over feature views** (0.7141 → 0.7172 → 0.7187 → 0.7221 → 0.7239):
   members differ in hyperparameters (depth 6–10, 30–500 trees, lr 0.03–0.1, colsample_bynode
   0.7–1.0, max_bin 512) *and* in the feature subset they see (raw / target-encoded / numeric /
   all). Single-model tweaks all sat in a 0.705–0.714 noise band; view-diverse averaging was the
   only reliable way past 0.715.
2. **Smoothed target encoding with out-of-fold values for train rows** (+~0.003 inside the
   ensemble): te_origin, te_dest, te_route, te_carrier, te_hour, te_dow, te_month, te_origin_hour,
   te_dest_hour (m = 20–200, fit on train only; 5-fold OOF encodings for the training matrix).
   Dense, year-stable keys transferred; sparse/date-level keys did not.
3. **Hubness/frequency features added to the TE and raw views** (0.7246 → 0.7257): log flight
   counts per Origin / Dest / Carrier / route — target-free, stable across the 2005→2006 shift.
4. **Pruning redundant raw categoricals** (0.7258 → 0.7270 → 0.7282): removing DayofMonth, then
   Month and DayOfWeek from the raw view (their smoothed TE counterparts cover them) gave the two
   largest late gains — raw members were wasting split budget on noisy 31/12/7-level categoricals.
5. **Temporal drift weighting** (0.7240 → 0.7245): sample weights ramped 0.5→2.0 across Jan→Dec
   2005, adapting members toward the 2006-like distribution of the eval/holdout period.

## Things that did not help

- **Finer/sparser target encodings**: te_date (Month|Day), te_carrier_hour, te_origin_dow,
  te_carrier_dow, te_dow_hour, te_dom — all encoded 2005-specific calendar noise; −0.0006 to −0.005.
- **Early stopping and row/column subsampling as single-model regularizers** (ES picked 109–193
  trees; subsample/colsample 0.8): every ES/subsample variant scored 0.7108–0.7120 vs 0.7141.
- **Rank-averaging vs probability averaging** of members (0.7219 vs 0.7221 — a tie), product-TE
  features (te_origin×te_dest), cyclical sin/cos departure time, DepHour as a categorical
  (also 2.6× slower due to categorical split search), and a 3rd seed batch of deep members.

## With more budget

Next steps would be: (1) proper two-level stacking — learn member/view blend weights on
out-of-fold 2005 predictions with shrinkage toward equal weights (plain averaging saturated at
~0.726; member count had stopped helping); (2) per-view hyperparameter tuning on the strong
(500-tree) members, which are the ensemble's backbone; (3) exploring route/airport-pair
interaction TEs with per-key adaptive smoothing; (4) testing whether hubness features computed
per (origin, hour) or carrier-airport TEs (m≈150) transfer like the origin/dest-hour TEs did.
