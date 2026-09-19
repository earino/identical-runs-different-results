#!/usr/bin/env python3
"""What the paper's findings look like on 2007 flights, a later test set nothing in this study had used.

Reads the scores from score_2007.py (every scored run of Studies 2 and 3, retrained and applied to the 2006 holdout
and to 1,000,000 flights from 2007) and prints:

  RESCORE      how closely retraining reproduces each run's recorded 2006 holdout AUC
  TRANSFER     the starting code and the delivered artifacts on both years, and how much of the agents' gain over the
               starting code survives into 2007; the rank correlation of runs' 2006 and 2007 scores inside a pairing
  SPREAD       Study 2's headline comparison on 2007: spread of the six pairing means against the median pairing SD
  POLICY       the repeat-and-select policy (chosen on the evaluation set, as in the paper) scored on 2007
  MODEL        Study 3's gain from the larger model on 2007, pooled and per agent
  RULES        where the rule-breaking runs land on 2007

Usage (repo root): python experiments/run_variance/analyze_2007.py [SCORES_CSV]   (data repository: analysis/)
"""
import csv, statistics as st, sys
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from best_of_k import pooled, quantile  # noqa: E402

import os
DATA_REPO = os.path.exists("task/airline/holdout.csv")
SCORES = sys.argv[1] if len(sys.argv) > 1 else ("data/2007/scores.csv" if DATA_REPO else "results/variance_2007/scores.csv")
FINAL = ({"study2": "data/study2/final_eval.csv", "study3": "data/study3/final_eval.csv"} if DATA_REPO else
         {"study2": "results/variance/final_eval.csv", "study3": "results/variance_glm53/final_eval.csv"})
CELLS = ({"study2": "data/study2/cells.csv", "study3": "data/study3/cells.csv"} if DATA_REPO else
         {"study2": "results/variance/cells.csv", "study3": "results/variance_glm53/cells.csv"})
START_2006, START_2007 = 0.7147707517540001, 0.717540430314   # the starting code (task_template/train.py), scored the same way
NAME = {"pi": "pi", "hermes": "Hermes", "opencode": "OpenCode"}

rows = {}
for r in csv.DictReader(open(SCORES)):
    if r["auc_2007"]:
        rows[(r["study"], r["harness"], r["model"], r["seed"])] = r
fe = {}
for study, path in FINAL.items():
    for r in csv.DictReader(open(path)):
        f = r["final_eval_auc"]
        fe[(study, r["harness"], r["model"], r["seed"])] = float(f) if f not in ("", "None") else 0.0
flags = {}
for study, path in CELLS.items():
    for r in csv.DictReader(open(path)):
        if r["counted"] == "True":
            flags[(study, r["harness"], r["model"], r["seed"])] = r
comp = {k: r for k, r in rows.items() if r["compliant"] == "True"}
missing = [k for k, r in flags.items() if r["status"] == "scored" and k not in rows]
print(f"scored on 2007: {len(rows)} runs ({len(comp)} compliant); not yet or not scorable: {len(missing)}")

print("\nRESCORE (retrained, applied to the 2006 holdout, against the recorded holdout AUC)")
for name, sub in (("compliant", comp.values()), ("broke a rule", [r for r in rows.values() if r["compliant"] != "True"])):
    d = sorted(abs(float(r["auc_2006_rescored"]) - float(r["holdout_auc"])) for r in sub)
    if d:
        print(f"  {name:13s} n {len(d):3d}  median |difference| {st.median(d):.5f}  95th percentile {d[int(0.95 * (len(d) - 1))]:.5f}  max {d[-1]:.5f}")

print("\nTRANSFER (compliant runs)")
by = defaultdict(list)
for k, r in comp.items():
    by[(k[0], k[1], k[2])].append((float(r["holdout_auc"]), float(r["auc_2007"])))
a06 = [x for v in by.values() for x, _ in v]; a07 = [y for v in by.values() for _, y in v]
print(f"  starting code: 2006 {START_2006:.4f}, 2007 {START_2007:.4f}")
print(f"  delivered, mean: 2006 {st.mean(a06):.4f}, 2007 {st.mean(a07):.4f}")
g06, g07 = st.mean(a06) - START_2006, st.mean(a07) - START_2007
print(f"  gain over the starting code: 2006 {g06:+.4f}, 2007 {g07:+.4f}  -> {g07 / g06:.0%} survives")
below = lambda v, s0: sum(x < s0 - 1e-6 for x in v); same = lambda v, s0: sum(abs(x - s0) <= 1e-6 for x in v)
print(f"  compliant runs below the starting code: 2006 {below(a06, START_2006)}, 2007 {below(a07, START_2007)} of {len(a07)} "
      f"(and {same(a07, START_2007)} that delivered it unchanged)")
xs, ys, per = [], [], []
for v in by.values():
    if len(v) < 5:
        continue
    h, t = [x for x, _ in v], [y for _, y in v]
    per.append(spearmanr(h, t)[0])
    rank = lambda z: list(np.argsort(np.argsort(z)) / (len(z) - 1))
    xs += rank(h); ys += rank(t)
print(f"  rank correlation of a run's 2006 and 2007 AUC inside its pairing: {np.corrcoef(xs, ys)[0, 1]:+.2f} "
      f"(pairings {min(per):+.2f} to {max(per):+.2f})")
for key in sorted(by):
    v = by[key]
    print(f"    {key[0]} {NAME[key[1]]:9s} {key[2]:19s} n {len(v):2d}  2006 {st.mean(x for x, _ in v):.4f}  "
          f"2007 {st.mean(y for _, y in v):.4f}  SD 2007 {st.stdev(y for _, y in v) if len(v) > 1 else 0:.4f}")

print("\nSPREAD (Study 2, compliant runs)")
for year, idx in (("2006", 0), ("2007", 1)):
    means = [st.mean(p[idx] for p in v) for k, v in by.items() if k[0] == "study2"]
    sds = [st.stdev(p[idx] for p in v) for k, v in by.items() if k[0] == "study2" and len(v) > 1]
    print(f"  {year}: pairing means span {max(means) - min(means):.4f}; median pairing SD {st.median(sds):.4f}")

print("\nPOLICY (Study 2; chosen on the evaluation set, scored on 2007)")
pairs = defaultdict(list)
for k, r in flags.items():
    if k[0] != "study2":
        continue
    ok = r["compliant"] == "True" and k in rows
    pairs[(k[1], k[2])].append((float(rows[k]["auc_2007"]) if ok else None, fe.get(k, 0.0), ok))
med = {kk: quantile(pooled(pairs, kk)[1], .5) for kk in (1, 3, 10)}
rng = np.random.default_rng(0); gains = {3: [], 10: []}
for _ in range(2000):
    boot = {key: [runs[i] for i in rng.integers(0, len(runs), len(runs))] for key, runs in pairs.items()}
    base = quantile(pooled(boot, 1)[1], .5)
    for kk in gains:
        gains[kk].append(quantile(pooled(boot, kk)[1], .5) - base)
ci = {kk: np.percentile(g, [2.5, 97.5]) for kk, g in gains.items()}
print(f"  median kept on 2007: one attempt {med[1]:.4f}, three {med[3]:.4f} ({med[3] - med[1]:+.4f}, 95% "
      f"{ci[3][0]:+.4f} to {ci[3][1]:+.4f}), ten {med[10]:.4f} ({med[10] - med[1]:+.4f}, 95% {ci[10][0]:+.4f} to {ci[10][1]:+.4f})")

print("\nMODEL (Study 3: GLM-5.3 against Study 2's GLM-5.3 Flash, compliant runs)")
for year, idx in (("2006", 0), ("2007", 1)):
    big = {h: [p[idx] for p in by.get(("study3", h, "glm-5.3"), [])] for h in NAME}
    fl = {h: [p[idx] for p in by.get(("study2", h, "glm-5.3-flash"), [])] for h in NAME}
    if not all(big.values()) or not all(fl.values()):
        print(f"  {year}: incomplete"); continue
    pool_b = [x for v in big.values() for x in v]; pool_f = [x for v in fl.values() for x in v]
    gain = st.mean(pool_b) - st.mean(pool_f)
    se = (st.variance(pool_b) / len(pool_b) + st.variance(pool_f) / len(pool_f)) ** .5
    per_agent = ", ".join(f"{NAME[h]} {st.mean(big[h]) - st.mean(fl[h]):+.4f}" for h in NAME)
    sd = st.median(st.stdev(v) for v in fl.values())
    print(f"  {year}: gain {gain:+.4f} ({gain / se:.1f} SE, {gain / sd:.2f} run-to-run SD, {16 * sd ** 2 / gain ** 2:.0f} runs per arm); {per_agent}")
    ch = {h: (st.mean(big[h]) - st.mean(fl[h]), st.variance(big[h]) / len(big[h]) + st.variance(fl[h]) / len(fl[h])) for h in NAME}
    for a_, b_ in (("pi", "hermes"), ("pi", "opencode"), ("hermes", "opencode")):
        d = ch[a_][0] - ch[b_][0]; sdd = (ch[a_][1] + ch[b_][1]) ** .5
        print(f"      {NAME[a_]} minus {NAME[b_]}: {d:+.4f} ({d / sdd:+.1f} SE)")

print("\nRULES (runs that broke a rule, on each year; rank among all scored runs of their study)")
for study in ("study2", "study3"):
    sub = [(k, r) for k, r in rows.items() if k[0] == study]
    for year, col in (("2006", "holdout_auc"), ("2007", "auc_2007")):
        order = sorted(sub, key=lambda kr: -float(kr[1][col]))
        top = [kr for kr in order[:10]]
        print(f"  {study} {year}: rule-breakers among the ten highest scores: {sum(kr[1]['compliant'] != 'True' for kr in top)}")
    ev = [(k, r) for k, r in sub if flags[k]["eval_trained"] == "True"]
    if ev:
        print(f"  {study} evaluation-label training, mean: 2006 {st.mean(float(r['holdout_auc']) for _, r in ev):.4f}, "
              f"2007 {st.mean(float(r['auc_2007']) for _, r in ev):.4f}  (n {len(ev)})")
    fs = [(k, r) for k, r in sub if flags[k]["frame_stats"] == "True"]
    if fs:
        print(f"  {study} batch features, mean: 2006 {st.mean(float(r['holdout_auc']) for _, r in fs):.4f}, "
              f"2007 {st.mean(float(r['auc_2007']) for _, r in fs):.4f}  (n {len(fs)})")
