# Final report — airline dep-delay classifier

**Best official Eval AUC: 0.7440** (experiment #32, commit `5001110`), up from the 0.7141 starting
baseline. `validate.sh` passes: `CONTRACT OK`, eval AUC via `predict_proba` on target-removed rows
= 0.7440. Final artifact: a 4-member XGBoost ensemble (all training inside `train.py`, all encoders
fit on `data/train.csv` only).

## Changes that mattered most

1. **Out-of-fold target encoding of hour-interactions** (+0.015): `te_route_hour`, `te_origin_hour`,
   `te_dest_hour`, `te_carrier_hour`, `te_hour`, built with count smoothing (m=50) and 5-fold OOF
   values for the training rows (no self-leakage). Time-of-day × route/carrier is the dominant
   signal in this dataset (delay rate rises from ~4% at 5am to ~74% at 9pm).
2. **Hierarchical route×hour TE** (+0.005): route×hour cells shrunk toward the *hour-level* prior
   (cell → hour-prior → global prior fallback). Encodes each route's deviation from the global
   daily curve, which transfers across years much better than raw cell means. A sparse
   origin×carrier×hour variant with the same hour-prior shrinkage added another +0.0006.
3. **Strong L1 regularization (`reg_alpha` 8→10)** (+0.005 over the whole tuning arc): with noisy
   TEs and a 2005→2006 shift, L1 pruning of weak features/splits generalized much better than
   depth/round tuning (alpha 2→8 was worth ~+0.01 in the single-model phase).
4. **Decorrelated multi-model ensemble** (+0.001–0.002): averaging probabilities of 4 XGB
   configs with different depth (5/6/6/5), alpha (10/14/13/8), colsample, lr, bin count and seeds.
   The key trick: each member is trained on a **different OOF fold assignment of the TEs**
   (KFold seeds 0/1/2) — decorrelating the dominant features added +0.0006 where same-param
   seed-averaging added nothing.
5. **Recency weighting of training rows** (+0.0004): sample weights `1 + ((month−1)/11)²`
   upweight late-2005 rows, which are closer in operational regime to the 2006 evaluation/holdout
   period.

## Things that did not help (measured, then reverted)

- **Plain main-effect TEs** (route, origin, dest, carrier alone) and dense interaction TEs
  (month×hour, dow×hour, route×dow): all neutral-to-harmful; the hour-interaction structure is
  where the signal is.
- **Stacking / residual features** (stage-1 OOF predictions as a stage-2 feature): the meta-model
  collapsed onto the stage-1 prediction and early-stopped almost immediately (0.7391).
- **Day-of-year cyclical features** (0.7400, −0.002) and depth-6-7 trees *without* stronger L1:
  let the model memorize 2005-specific seasonal/interaction patterns that didn't transfer to 2006.
  Also neutral: window-TEs borrowing from adjacent hours, distance-deviation features, congestion
  counts, subsampling, rank-averaging, and a 5th ensemble member (wall-clock limit).

## With more budget

The eval plateau for a single model was ~0.741–0.742 and ensembling reliably bought +0.002; I'd
push the ensemble further (6–8 members with distinct fold seeds and TE smoothing levels), which
needs a leaner per-member fit to stay under the 120 s cap (e.g. lr 0.05 with patience 60). Second,
I'd attack the mid-day AUC dip (0.65 at 12–17h vs 0.87 early morning): the raw features say little
about *why* a 2pm flight is late; more granular scheduled-time structure (30-min slot TEs with
adaptive smoothing) or origin×carrier×slot curves are the best remaining shot. Third, I'd make
the 2005→2006 alignment more principled: recency-weight the TE group statistics themselves (tried
once, eval-neutral) and explore exponential decay, plus two-pass CV where early-stopping uses a
held-out 2005 quarter instead of the 2006 eval slice to avoid any eval-set overfitting in the
final holdout.
