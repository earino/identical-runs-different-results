#!/usr/bin/env python3
"""The Study 2 and 3 numbers the paper quotes that no other script prints.

  spread       span of the six compliant pairing means; median pairing SD (compliant and all runs); gaps between two
               runs of one pairing; the weakest compliant run against the starting code
  three runs   how often the weaker pairing wins when each pairing contributes three compliant runs, drawn with
               replacement; exact, by comparing every possible triple of one pairing with every triple of the other
  run counts   16 sd^2 / gap^2 runs per arm, for the gaps the text names, and the Study 3 effects table
  compute      rank correlation between budget use and score inside pairings, with subsets by budget share
  same day     Study 1's GLM-5.3 and GLM-5.3 Flash rows, which ran on the same day, overlapping in time: the gain for
               Study 3's three agents with no day boundary between the arms

--legacy reproduces the numbers under the exclusion rule the paper used before 18 September 2026 (the evaluator's
refit_suspect screen or frame statistics), as a check that this script matches the original method.

Usage (repo root): python experiments/run_variance/paper_numbers.py [--legacy] [S2_CELLS S3_CELLS [S1_CELLS]]
"""
import csv, itertools, math, random, statistics as st, sys
from collections import defaultdict

import numpy as np

args = [a for a in sys.argv[1:] if not a.startswith("--")]
LEGACY = "--legacy" in sys.argv
S2 = args[0] if args else "results/variance/cells.csv"
S3 = args[1] if len(args) > 1 else "results/variance_glm53/cells.csv"
S1 = args[2] if len(args) > 2 else "results/phase2/merged/cells.csv"
BUDGET = 18000.0
T = lambda v: v == "True"
ok = (lambda r: not T(r["refit_suspect"]) and not T(r["frame_stats"])) if LEGACY else (lambda r: T(r["compliant"]))


def load(path):
    return [r for r in csv.DictReader(open(path)) if T(r["counted"]) and r["status"] == "scored" and r["holdout_auc"]]


s2, s3 = load(S2), load(S3)
base = float(s2[0]["baseline_auc"])
comp = defaultdict(list); allr = defaultdict(list)
for r in s2:
    allr[(r["harness"], r["model"])].append(float(r["holdout_auc"]))
    if ok(r):
        comp[(r["harness"], r["model"])].append(float(r["holdout_auc"]))
means = {k: st.mean(v) for k, v in comp.items()}
sds = {k: st.stdev(v) for k, v in comp.items()}
print(f"rule: {'legacy (refit_suspect or frame_stats)' if LEGACY else 'compliant column'}; Study 2 compliant runs {sum(map(len, comp.values()))}")

print("\nSPREAD")
print(f"  compliant pairing means span {max(means.values()) - min(means.values()):.4f}; median pairing SD {st.median(sds.values()):.4f} "
      f"(all runs {st.median(st.stdev(v) for v in allr.values()):.4f})")
diffs = sorted(abs(a - b) for v in allr.values() for a, b in itertools.combinations(v, 2))
print(f"  two runs of one pairing, all runs: mean gap {st.mean(diffs):.4f}, ninth decile {diffs[int(0.9 * len(diffs))]:.4f}")
print(f"  weakest compliant run beats the starting code by {min(min(v) for v in comp.values()) - base:.4f}")
# the holdout's own sampling error: Hanley and McNeil (1982), for an AUC near the observed one on 500,000 positives and
# 500,000 negatives. Rows are treated as independent draws from the holdout's population (2006 flights).
A, n1, n0 = st.median(float(r["holdout_auc"]) for r in s2), 500000, 500000
q1, q2 = A / (2 - A), 2 * A * A / (1 + A)
print(f"  holdout AUC standard error (Hanley-McNeil, AUC {A:.4f}, 500k per class): "
      f"{math.sqrt((A * (1 - A) + (n1 - 1) * (q1 - A * A) + (n0 - 1) * (q2 - A * A)) / (n1 * n0)):.5f}")

print("\nTHREE RUNS (compliant, drawn with replacement as in best-of-k; exact)")
shares = []


def triple_sums(v):
    v = np.array(v)
    return np.sort((v[:, None, None] + v[None, :, None] + v[None, None, :]).ravel())


for model in sorted({k[1] for k in comp}):
    ks = [k for k in comp if k[1] == model]
    for a, b in itertools.combinations(ks, 2):
        weak, strong = sorted((a, b), key=lambda k: means[k])
        w, s_ = triple_sums(comp[weak]), triple_sums(comp[strong])
        share = np.searchsorted(s_, w, side="left").sum() / (len(w) * len(s_))   # strong triple strictly below weak
        shares.append(share)
        print(f"  {model:19s} {weak[0]:9s} (weaker) beats {strong[0]:9s} {share:.1%}")
print(f"  range {min(shares):.0%} to {max(shares):.0%}")

print("\nRUN COUNTS (16 sd^2 / gap^2 per arm, sd = median compliant pairing SD)")
sd = st.median(sds.values())
widest = max(means.values()) - min(means.values())
same = max(max(means[k] for k in means if k[1] == m) - min(means[k] for k in means if k[1] == m) for m in {k[1] for k in means})
for name, gap in (("widest gap among the six pairings", widest), ("widest gap between two agents on one model", same)):
    print(f"  {name}: {gap:.4f} -> {16 * sd ** 2 / gap ** 2:.1f} runs of each")

print("\nSTUDY 3 EFFECTS TABLE (sd = median GLM-5.3 Flash pairing SD)")
flash = defaultdict(list); big = defaultdict(list)
for r in s2:
    if r["model"] == "glm-5.3-flash" and ok(r):
        flash[r["harness"]].append(float(r["holdout_auc"]))
for r in s3:
    if ok(r):
        big[r["harness"]].append(float(r["holdout_auc"]))
sdf = st.median(st.stdev(v) for v in flash.values())
pooled = lambda d: st.mean([x for v in d.values() for x in v])
both = st.mean([x for d in (flash, big) for v in d.values() for x in v])
rows = [("did the agent improve on the starting code?", both - base),
        ("did the larger model improve the pairing?", pooled(big) - pooled(flash)),
        ("are two agents different on GLM-5.3?", max(map(st.mean, big.values())) - min(map(st.mean, big.values()))),
        ("are two agents different on GLM-5.3 Flash?", max(map(st.mean, flash.values())) - min(map(st.mean, flash.values())))]
print(f"  sd {sdf:.4f}")
for q, e in rows:
    print(f"  {q:45s} {e:+.4f}  {e / sdf:4.2f} SD  {16 * sdf ** 2 / e ** 2:6.1f} runs per arm")

print("\nSAME-DAY CHECK, Study 1 (both GLM models on 13 September, overlapping in time; compliant runs of pi, Hermes, OpenCode)")
s1 = [r for r in csv.DictReader(open(S1)) if r["model"] in ("glm-5.3", "glm-5.3-flash") and r["harness"] in ("pi", "hermes", "opencode")
      and r["holdout_auc"] and (r["n_experiments"] or "0") not in ("0", "0.0") and r["refit_suspect"] != "True"]
g = {m: [float(r["holdout_auc"]) for r in s1 if r["model"] == m] for m in ("glm-5.3", "glm-5.3-flash")}
d = st.mean(g["glm-5.3"]) - st.mean(g["glm-5.3-flash"])
se = math.sqrt(st.variance(g["glm-5.3"]) / len(g["glm-5.3"]) + st.variance(g["glm-5.3-flash"]) / len(g["glm-5.3-flash"]))
print(f"  GLM-5.3 {st.mean(g['glm-5.3']):.4f} ({len(g['glm-5.3'])} runs)   GLM-5.3 Flash {st.mean(g['glm-5.3-flash']):.4f} "
      f"({len(g['glm-5.3-flash'])} runs)   gain {d:+.4f}  ({d / se:.1f} SE)")

print("\nCOMPUTE AGAINST SCORE, Study 2 (ranks within pairing, pooled; bootstrap 2,000; two-sided permutation within pairing 2,000)")


def within_rank_corr(rows):
    """Spearman correlation of budget share and score, each ranked inside its own pairing, then pooled."""
    by = defaultdict(list)
    for r in rows:
        by[(r["harness"], r["model"])].append(r)
    xs, ys = [], []
    for grp in by.values():
        if len(grp) < 3:
            continue
        rank = lambda vals: [sorted(vals).index(v) / (len(vals) - 1) for v in vals]
        xs += rank([share(r) for r in grp]); ys += rank([float(r["holdout_auc"]) for r in grp])
    mx, my = st.mean(xs), st.mean(ys)
    cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    return cov / math.sqrt(sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys))


share = lambda r: (float(r["cpu_seconds_counted"] or 0) + float(r["cpu_seconds_uncounted"] or 0)) / BUDGET
withcpu = [r for r in s2 if r["cpu_seconds_counted"]]
rng = random.Random(0)
subsets = [("all runs", withcpu), ("compliant runs", [r for r in withcpu if ok(r)])]
for cut, label in ((0.25, "a quarter"), (0.5, "half"), (0.75, "three quarters")):
    subsets.append((f"compliant, above {label} of budget", [r for r in withcpu if ok(r) and share(r) > cut]))
for name, rows_ in subsets:
    rho = within_rank_corr(rows_)
    boots = sorted(within_rank_corr([rng.choice(rows_) for _ in rows_]) for _ in range(2000))
    by = defaultdict(list)
    for r in rows_:
        by[(r["harness"], r["model"])].append(r)
    perm_ge = 0
    for _ in range(2000):
        shuffled = []
        for grp in by.values():
            scores = [r["holdout_auc"] for r in grp]; rng.shuffle(scores)
            shuffled += [dict(r, holdout_auc=s) for r, s in zip(grp, scores)]
        perm_ge += abs(within_rank_corr(shuffled)) >= abs(rho)
    print(f"  {name:38s} n {len(rows_):3d}  rho {rho:+.2f}  95% {boots[50]:+.2f} to {boots[1949]:+.2f}  p {perm_ge / 2000:.3f}")
print(f"  runs at or under a quarter of the budget: all {sum(share(r) <= 0.25 for r in withcpu)}, compliant "
      f"{sum(share(r) <= 0.25 for r in withcpu if ok(r))}")
