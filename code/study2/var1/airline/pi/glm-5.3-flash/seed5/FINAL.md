# Final Report — airline delay AUC

**Best Eval AUC: 0.7341** (baseline 0.7141 → +0.020). Final commit `21ff413`, validation: `CONTRACT OK`.

## Model
16-member XGBoost ensemble (hist, `enable_categorical`), probability-averaged:
depths 3–8 + lossguide variants (leaves 64–256), lr 0.03, subsample 0.85, colsample 0.6–0.9,
mcw 5, reg_lambda 1.0; plus 4 feature-subset members (TE-only / base-only). Trees scaled ×1.5–2
per member (shallow members get more). Two target-encoding fold-seed variants alternate across
members to decorrelate TE noise.

## Features (all inside `prepare()`, encoders fitted on train only)
- `hour = DepTime // 100` (+ minute, integer-encoded dates, distance, 3 native categoricals)
- 9 smoothed OOF target encodings (KFold-5): carrier/origin/dest (m 30–60), carrier×hour,
  origin×hour, dest×hour (m 150), dow×hour (m 150), route (m 300), origin×dow (m 100)
- 4 congestion-volume counts on train: origin×hour, dest×hour, origin×dow, route (`fillna(0)`)

## What mattered most
1. **DepTime hour** — delay rate spans ~4% (5am) → ~74% (9pm); the single dominant signal.
2. **OOF target encoding** of categorical × hour interactions (+0.0015 over native cats alone).
3. **Ensembling diverse members** (2 → 16): +0.0036 cumulative; feature-subset members helped.
4. **Congestion-volume counts** (+0.0017 with 16 members) — flights/hour at the origin/dest.
5. **More trees per member** (×1.5–2 at fixed lr) (+0.0009–0.0015).

## What did not help
- Early stopping on a Nov–Dec time split (seasonality mismatch; died at ~42 trees).
- Stacking/meta-learner on OOF predictions, OOF-AUC-weighted averaging, rank objectives.
- Freq/count encodings, log-distance, route as native categorical, TE on month, hour wrap-24,
  depth 8–9 single models, max_bin 64/512, lr 0.02/0.04, 20 members at ×1 trees, TE×volume products.

## With more budget
Push the member count to ~24 with per-member TE fold seeds and shallow-member tree scaling
(two cheap wins that stopped only on CPU/timeout limits), then tune the volume features
(hour-window counts, relative congestion = vol ÷ daily mean), and re-test stacking with the
richer feature set. Gains were flattening near 0.734 — remaining headroom is likely in
interaction features (origin×dow×hour volumes) rather than model capacity.
