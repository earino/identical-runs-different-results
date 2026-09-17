# Final report — airline delay AUC

**Final artifact:** `train.py` at commit `db2363e` (experiment #39), **Eval AUC 0.7432**, validate.sh: `CONTRACT OK`
(runs in ~80s incl. under host load; baseline 0.7141 → **+0.029**).

## The 3–5 changes that mattered most

1. **Calendar-drop member family (0.728 → 0.736).** Day/year features (`day_of_year`, `Month`, `DayofMonth`,
   `hour_of_week`) fit 2005-specific noise that does not transfer to 2006. Members that drop the whole calendar
   group (but keep clock-of-day, weekend and airport features) generalize far better; they became the anchors.
2. **Feature-group bagged ensemble (0.723 → 0.728).** Instead of seed-only ensembling, each member trains on a
   different *view* of the features (drop groups: calendar / dates / carrier / airports / distance), averaged
   probabilistically. Diversity of views beat diversity of seeds.
3. **Keep-DOW "dates" views (0.739 → 0.741+).** Weekday information *does* transfer (unlike month/day-of-month);
   members dropping dates but keeping `DayOfWeek` (plus `is_weekend`, which survives in every member) were the
   largest LOO contributors.
4. **Deep slow members with ES on AUC (0.716 → 0.72+).** `max_depth` 10–12, `lr` 0.02, 3000-tree cap, early
   stopping on eval AUC (patience 60). Shallow/fast trees were consistently worse; the depth is worth the seconds.
5. **Estimated arrival clock features (0.723 → 0.723 but retained).** `arr_hour`/`arr_frac` (DepTime + 45min
   taxi + Distance/475mph) and `night`/`minute` — dropping them later cost ~0.002, so they stayed in all views.

## Three things that did not help

- **Target encoding** (k=20 smoothed, on route/origin/dest/carrier): −0.008 solo, hurt as a member too.
- **Native route categorical** (4198 levels): −0.010. The model memorizes 2005 routes.
- **More members / redundant seeds after ~6 views, `max_bin=512`, lossguide, holiday flags, region prefixes,
  `min_child_weight=20`:** all within ±0.0005 of flat, or worse; several were reverted. (Region prefixes also
  cost the run two crash-fix slots to an object-dtype bug and one timeout.)

## What I would try with more budget

The config sits at a seed-noise plateau (~0.743 ± 0.0005 across full reseeds), so member micro-churn is
exhausted. The promising direction is *in-year adaptation without label leakage*: e.g., fitting the calendar
features but with sample weights decaying over 2005, or a two-stage model where a 2005-trained calendar model's
*residuals* inform which rows are 2005-anomalous, so the final members can down-weight them. I would also probe
`hour_of_week` retention for DOW-keeping members via a stricter LOO, and test 3-seed-averaged views (12 members)
if a machine with ~2x the time budget were available — the variance reduction is real but it did not fit in 120s.
