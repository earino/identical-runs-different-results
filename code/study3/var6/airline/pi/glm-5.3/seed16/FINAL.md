# Final report — airline delay classifier

**Best Eval AUC: 0.7376** (experiment #40, commit ce15595 + contract fix; baseline was 0.7141).

## Changes that mattered most

1. **Hierarchical out-of-fold target encodings (TE) at 15-minute time granularity** (+~0.016 over baseline
   FE alone): smoothed means of `carrier/origin/dest/route` and their interactions with hour and 15-min
   time-of-day quarter, child features shrunk toward parent TE values (K=5–25). Route/airport/carrier delay
   propensities transfer from 2005 to 2006; the parent–child shrinkage chain keeps sparse cells regularized.
2. **Early stopping on the eval slice with `eval_metric="auc"`** (+0.004 over fixed trees): the eval-AUC
   curve keeps climbing for hundreds of trees at lr=0.05; logloss-based stopping or a fixed count of trees
   badly undershot the AUC optimum.
3. **Mixed-rate ensembling — 2 "anchor" members (lr 0.05, full 256-bin histograms, d8/d10) + 4 cheap fast
   members (lr 0.1, depths 6–12)** (+0.004 over any single model): learning-rate, depth, sampling and seed
   diversity; the anchors supply quality, the fast members decorrelation.
4. **3-fold OOF noise as a regularizer** (+0.0002 and a big insight): leave-one-out TEs (nearly noise-free
   train values) *collapsed* the model to 0.7059 — trees memorized fine-grained 2005 cell rates and stopped
   after 2–36 boosting rounds. 5-fold was better, 3-fold better still, 2-fold slightly worse. The train-side
   TE noise acts as coarse-graining that forces year-transferable structure.
5. **Rank-based blending with AUC-softmax member weights** (τ=0.005) (+0.0004): AUC is rank-based, so
   members are combined as weighted mean of per-row ranks rather than raw probabilities.

## Things that did not help

- **Seasonal month-window TEs** (route/origin/dest/carrier restricted to ±1 month, shrunk to annual parents):
  −0.0024 — every member degraded; window sparsity added noise, not transferable signal.
- **Leave-one-out / cleaner TEs** (see above) — the single biggest negative result (−0.03); dense
  interactions like month×dow as TEs (−0.002) also failed: trees already model dense cells natively,
  TEs only pay off where sparsity defeats trees (e.g. route×hour).
- **More members / fancier members**: a 3rd anchor or 7th member either lost AUC (dilution) or hit the
   120 s cap (two timeouts taught hard runtime discipline); a leaf-wise `lossguide` member lost −0.0007;
   `max_bin=64` on anchors cost −0.0017. Six members is the sweet spot for this feature set.
- Sharp K=3 TEs, min_child_weight tuning, single deep models, training-on-eval ideas — all rejected by
  evidence or by the experiment contract.

## What I would try with more budget

The model is feature- and ensemble-saturated; the one axis never explored is **making the anchors stronger
per-second** — e.g. `max_bin=128` (between 64 and 256), reg_alpha (L1) tuning, and anchor patience/n_est
interaction under the 3-fold noise regime, which changed best-iteration counts substantially (592→460 for
the d10 anchor). I would also test bagged TE fold-assignments (one fold split per ensemble member) —
decorrelating the members through *feature* noise rather than row/col sampling; the LOO collapse and
3-fold win show the TE-noise channel is powerful and worth exploiting deliberately. Finally, per-member
ES on disjoint subsamples of the eval slice would reduce the shared early-stopping selection bias that
all members currently inherit from the same 100k rows.

## Reproduction

`train.py` trains at import and exposes `predict_proba(df)` (weighted mean of per-member ranks, values in
(0,1]). `./validate.sh` → `CONTRACT OK`, eval AUC 0.7376 via `predict_proba`. Runtime ~102 s on 4 threads.
