# Final Report

**Best Eval AUC: 0.7186** (experiment #20, commit acb5da1 — baseline was 0.7141)

## What mattered most (top 3-5)

1. **`carrier × hour` categorical interaction** (E20, +0.0045 → 0.7186). The single largest gain.
   Carriers have very different delay profiles at different times of day; XGBoost alone could not
   capture this from raw categorical × numeric splits.
2. **`origin × hour` and `dest × hour`** (E21, +0.0028 on top of baseline but −0.003 on top of E20).
   Directionally useful (airport congestion is hour-dependent) but overlaps with carrier-hour signal.
3. **Early stopping on an internal 15% holdout from train** (E2–E10 line of work). Stopped overfitting
   to 2005 data and let us use more trees; not by itself a gain, but necessary headroom.
4. **Keeping max_depth=6, lr=0.1, subsample/colsample=0.9.** Deeper trees (8/10/12) all hurt badly;
   the signal is broad and shallow, not deep-and-interactive.

## What did not help

1. **`route = Origin_Dest` categorical** (E13, 0.7039) — too sparse, mostly unseen in eval/holdout.
2. **`carrier × month`, `carrier × day-of-month`, `month × day-of-month`** (E14–E17) — all worse than
   baseline; the delay pattern is not seasonal by carrier in a way that generalizes from 2005 to 2006.
3. **Deeper trees** (E3 depth10 = 0.7076, E6 depth12 = 0.7039) — severe overfit.
4. **Triple interactions** (`carrier×month×dom`, `origin×month×dom`, etc., E35 = 0.6888) — pure noise.
5. **`origin × hour × dow` / `carrier × hour × dow` / `dest × hour × dow`** (E31, E32, E33, E34) — all
   landed at ~0.7144, i.e. no better than baseline, at much higher compute cost.

## What I would try with more budget

- **Better time encoding**: `DepTime` is hhmm as a plain integer; a proper cyclical encoding
  (sin/cos of the minute-of-day, with the 2400→0 wrap) might let the model learn smooth time effects
  instead of 25 hard buckets per carrier.
- **Target/count encoding** of Origin and Dest with smoothing, instead of pure categorical splits —
  especially useful for rare airports.
- **Smarter interaction search**: exhaustively testing pairs was slow and mostly negative; a
  greedy "add the one interaction that improves internal-holdout AUC most, then re-check" loop with
  early stopping would find the right 1–2 interactions instead of guessing.
- **Ensembling** 3–5 XGBoost models with different seeds/feature subsets (allowed by the contract,
  all XGBoost) — a straightforward +0.001–0.002 typically.
