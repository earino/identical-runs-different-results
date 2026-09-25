#!/usr/bin/env python3
"""Stage-1 decision table: is the between-agent spread larger than run-to-run (seed) noise?

Reads a merged cells.csv with a `seed` column (MERGE_OUT=results/phase2/merged scripts/merge_report.py pulls/phase2_*/runs).
For every (dataset, provider, model, harness) pair: n seeds, mean, min, max, range of holdout AUC. For every model row:
the between-agent range of pair means vs the median within-pair seed range, and the ratio. Also the budget's own
audit per pair: mean budgeted CPU s, refusals, kills, uncounted fits. Usage:
  .venv/bin/python scripts/phase2/stage1_analysis.py [results/phase2/merged/cells.csv]
"""
import math
try:
    from statistics import NormalDist
except ImportError:  # pragma: no cover
    NormalDist = None
def ci95(vals):
    """mean, SE, 95% half-width (t-based for n<30; z for large n). Returns (mean, se, hw) or (mean, None, None) for n<2."""
    n = len(vals)
    if n < 2:
        return (sum(vals)/n if n else float("nan")), None, None
    m = sum(vals)/n; sd = (sum((v-m)**2 for v in vals)/(n-1))**0.5; se = sd/n**0.5
    t = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 7: 2.45, 8: 2.36, 9: 2.31, 10: 2.26, 11: 2.23, 12: 2.20, 15: 2.14, 20: 2.09, 25: 2.06}
    tv = t.get(n) or (2.14 if n < 20 else (2.09 if n < 25 else 2.0))
    return m, se, tv*se
import csv, statistics as st, sys
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "results/phase2/merged/cells.csv"
f = lambda x: float(x) if x not in ("", "None", None) else None
allrows = list(csv.DictReader(open(path)))
excluded = [r for r in allrows if r.get("provider_error") or r.get("refit_suspect") == "True" or f(r.get("holdout_auc")) is None
            or int(f(r.get("n_experiments")) or 0) == 0]   # a cell that never ran an experiment delivers the baseline, not a result
rows = [r for r in allrows if r not in excluded]
print(f"{len(rows)} cells in; {len(excluded)} excluded (provider_error, refit-suspect, or unscored):")
for r in excluded: print(f"   - seed{r.get('seed')} {r['dataset']} {r['provider']} {r['model']} {r['harness']}: "
                         f"{'provider_error' if r.get('provider_error') else ''}{'refit-suspect (eval %s vs holdout %s)' % (r.get('best_eval_auc'), (r.get('holdout_auc') or '')[:6]) if r.get('refit_suspect')=='True' else ''}{'unscored' if f(r.get('holdout_auc')) is None else ''}{'no experiments (delivered the baseline)' if int(f(r.get('n_experiments')) or 0) == 0 else ''}")
pairs = defaultdict(list)
for r in rows:
    pairs[(r["dataset"], r["provider"], r["family"], r["harness"])].append(r)

print(f"{'dataset':8} {'provider':9} {'model':24} {'agent':9} n  mean    ±95%CI  min     max     range   cpu_bud  refus kill fits_unc")
summary = {}
for k in sorted(pairs):
    v = pairs[k]; a = [f(r["holdout_auc"]) for r in v]
    cpu = [f(r.get("cpu_seconds_budgeted")) for r in v if f(r.get("cpu_seconds_budgeted")) is not None]
    refus = sum(int(f(r.get("cpu_refusals")) or 0) for r in v); kill = sum(1 for r in v if r.get("cpu_killed") == "True")
    unc = sum(int(f(r.get("fits_uncounted")) or 0) for r in v)
    summary[k] = (st.mean(a), max(a) - min(a), len(a))
    _, _, hw = ci95(a)
    print(f"{k[0]:8} {k[1]:9} {k[2]:24} {k[3]:9} {len(a)}  {st.mean(a):.4f}  {('%.4f' % hw) if hw is not None else '   -  '}  {min(a):.4f}  {max(a):.4f}  {max(a)-min(a):.4f}  "
          f"{(st.mean(cpu) if cpu else float('nan')):7.0f}  {refus:5d} {kill:4d} {unc:8d}")

print("\n=== per model row: between-agent range of means vs within-pair seed range")
print(f"{'dataset':8} {'provider':9} {'model':24} agents  between  within_med  within_max  ratio  verdict")
by_row = defaultdict(list)
for k, (m, rng, n) in summary.items():
    by_row[k[:3]].append((k[3], m, rng, n))
for row in sorted(by_row):
    v = by_row[row]; means = [m for _, m, _, _ in v]; within = [rng for _, _, rng, n in v if n >= 2]
    between = max(means) - min(means)
    ses = [ci95([f(r["holdout_auc"]) for r in pairs[row + (h,)]])[1] for h, _, _, n in v if n >= 2]
    ses = [x for x in ses if x is not None]
    mdd = 2.8 * st.median(ses) if ses else None   # ~ smallest between-agent mean difference resolvable at 95% with these n
    if within:
        wmed, wmax = st.median(within), max(within); ratio = between / wmed if wmed else float("inf")
        verdict = "agent spread >> seed noise" if ratio >= 3 else ("comparable: replicate more" if ratio >= 1.5 else "noise dominates")
        print(f"{row[0]:8} {row[1]:9} {row[2]:24} {len(v):6d}  {between:.4f}   {wmed:.4f}      {wmax:.4f}     {ratio:5.1f}  {verdict}"
              + (f"  | min detectable diff ≈ {mdd:.4f}" if mdd else ""))
    else:
        print(f"{row[0]:8} {row[1]:9} {row[2]:24} {len(v):6d}  {between:.4f}   (single seed, no within-pair range)")
