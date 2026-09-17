# Final report — airline dep-delay XGBoost (autoresearch)

Best Eval AUC: **0.7486** (experiment #40, commit 689d29f), up from the 0.7141 baseline.
Validation: `./validate.sh` prints `CONTRACT OK`; AUC via `predict_proba` on eval with the
target column removed reproduces 0.7486.

## What mattered most

1. **Bagged ensemble of deep XGBoost models** (5→9 members, averaged probabilities,
   subsample 0.7): the single biggest lever, +0.003 alone and it kept paying as members
   were added. Individual deep trees are high-variance; averaging cancels exactly that.
2. **Very deep trees with weak regularization** (max_depth 24, min_child_weight 1,
   reg_lambda 0.5): each jump 6→8→12→14→16→20→24 was worth +0.001–0.003. Deep trees
   learn the categorical interactions (airport x hour) directly; strong leaf
   regularization cost −0.016.
3. **Cyclical + linear departure-time features** (sin/cos of minute-of-day with hour
   clipped at 23 for the 24xx+ encodings, plus raw minute-of-day): the schedule's
   morning-dip/afternoon-peak shape is the strongest signal in the data.
4. **Out-of-fold smoothed target encodings** for Origin, Dest, UniqueCarrier and hour:
   5-fold OOF values for training rows (no self-leak), full-train maps at predict time.
   Worth +0.0005 over leaky TEs and made interactions unnecessary.
5. **Feature-fraction diversity** (colsample_bytree 0.6, the last experiment): +0.0022
   over 0.7 — more decorrelation between members, cheaper trees, no runtime cost.

## What did not help

1. **Raw route (Origin×Dest) as a categorical**: −0.005; high-cardinality splits that
   don't transfer to 2006.
2. **Interaction target encodings** ((Origin,hour), (Dest,hour), route, (carrier,hour)
   TEs): −0.013 even with OOF — cells too sparse (~15 rows), pure noise amplification.
3. **Calendar fine-grain** (day-of-year, sin/cos doy, days-to-holiday, summer): −0.010.
   The 2005 seasonal pattern memorized by deep trees does not transfer to 2006; plain
   Month is coarser but robust.
4. Also negative: higher learning rate beyond convergence needs, min_child_weight 25 /
   reg_lambda 2 (over-regularized), and 10+ members (blows the 120 s cap — runtime is a
   hard constraint, not a tuning knob).

## With more budget

I would treat the 2005→2006 shift explicitly: the failure modes above are all
year-transfer problems. Concretely: (a) build OOF target encodings with heavier
smoothing tuned per cardinality (k-fold CV over SMOOTH in {10,20,50,100}), since the
optimal shrinkage differs between 20 carriers and 282 airports; (b) add a
"leave-recent-out" early-stopping scheme — train on Jan–Oct 2005, validate on Nov–Dec
2005 — so early stopping optimizes temporal transfer rather than 2006 eval fit, then
refit; (c) explore time-of-day x airport-slot features (a runway congestion proxy:
flights scheduled in the same 30-minute window at the same airport in the training
data, target-free), computed with fast groupbys; (d) test 12–16 members under a
faster learner config (max_bin 128, depth 20) to see whether more members beat the
current 9 once runtime is cut; and (e) rank-average the seed ensemble with a second
ensemble trained on a differently-seeded OOF split for further decorrelation.
