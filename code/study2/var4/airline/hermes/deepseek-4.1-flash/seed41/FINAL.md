# FINAL — airline delay (AUC)

**Best Eval AUC: 0.7377** (8-member XGBoost bag; baseline 0.7141, +0.0236).
40/40 experiments used, ~85 min wall clock, ~5.1k of 18k CPU-seconds. `validate.sh` → `CONTRACT OK`
(`predict_proba` reproduces 0.7377 on `data/eval.csv` with the target column dropped).

## What mattered most

1. **Departure-time features** (`dep_hour`, `dep_minute`, `dep_tod`, cyclic `tod_sin/cos`). The
   per-hour delay rate is the dominant signal (0.04 at 05:00 → 0.82 at 23:00 in both years). This single
   step took the baseline 0.7141 → 0.7201.
2. **Arrival-time proxy** (`arr_tod = dep_tod + 60*Distance/500`, plus its cyclic pair). Destination-side
   congestion at the estimated arrival time; +0.0020 (0.7315 → 0.7335), and the follow-up "destination
   traffic at the arrival hour" (`dest_hour_arr`, `dest_hour_arr_frac`) added +0.0014.
3. **Train-derived traffic-volume / hour-share features**: `orig_freq`, `orig_routes`,
   `carrier_orig_frac` (hub dominance), `orig_hour`, `carrier_hour`, `dest_hour` and their
   `*_frac` ratios. Aggregates computed on 2005 transfer to 2006; +0.0015, +0.0029, +0.0005, +0.0005 across steps.
4. **Regularization first, then a slow-learning full-data bag**: `min_child_weight=30`,
   `colsample_bytree≈0.7`, `subsample=0.85`, `reg_lambda=2`, `lr=0.05`. Only once trees were regularized
   did lr 0.05/2000 trees win (0.7250 → 0.7279); a 3-member bag over 100% of the rows at a fixed 900
   trees (tree count learned from an early-stopping study) gave 0.7310, and 5 → 7 → 8 diverse members
   reached 0.7338 → 0.7355 → 0.7377.
5. **Deleting calendar features**: dropping the month/day-of-month/day-of-week cyclic encodings
   (+0.0014) and the raw `Month` column (+0.0005) improved the score *and* removed code — 2005 seasonality
   does not transfer to 2006, while time-of-day and airport/carrier structure do. `DayofMonth` and
   `DayOfWeek` (raw) do carry transferable signal (removing them cost −0.0022 and −0.0035).

## What did not help

1. **Fine-grained identity/schedule features**: a `Route` (Origin_Dest, 4.2k levels) categorical cost
   −0.0163 (0.7377 → 0.7044 when tried early), and exact-slot congestion (`origin|DepTime`,
   `carrier|DepTime`) cost −0.0090. Both memorise 2005 schedules that change by 2006.
2. **Out-of-fold (smoothed, 5-fold) target encoding** of Origin/Dest/UniqueCarrier: −0.0007 — the hidden
   2006 slice shifts every level mean slightly.
3. **Extra capacity without extra regularization**: lr 0.05→0.025 with 5000 trees (−0.0005), depth 8
   (−0.0008), 1600 trees per member (−0.0011), depth 6 (−0.0013), mcw 60/λ4 (−0.0026), day-of-year
   cyclics + weekend flag (−0.0014), `max_bin=512` (neutral), logit- vs probability-averaging in the bag
   (neutral), 3-hour congestion windows (neutral, −0.0002).

## With more budget

The clearest untried lever is **stacking an XGBoost meta-learner over out-of-fold member predictions**
(the contract allows only XGBoost, but a level-2 XGBoost is still XGBoost): 8 members × 4 folds is ~4× the
current 85 s fit, so it needs a trimmed member set to stay under the 120 s cap, and it would let the
ensemble learn *where* each member is reliable instead of a uniform average. Second, **drift-aware
training**: adversarial validation between 2005 and 2006 rows to detect and either drop or down-weight the
features on which the two years disagree most (my calendar-feature results suggest this would find the
same family of features automatically), plus sample weighting toward calendar slices whose covariate
profile matches the target year. Third, an **AUC-aligned objective** (`rank:pairwise`, or `eval_metric`
with early stopping on AUC plus a PR-aware variant) instead of logloss, which is currently only a proxy.
Fourth, refine the arrival proxy — 500 mph is a rough block-speed constant; without real scheduled
arrival times, per-distance-band speeds fitted on the training year would sharpen the destination
congestion signal that produced the largest late gains.
