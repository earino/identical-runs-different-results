#!/usr/bin/env python3
"""Agent x model interaction as difference-in-differences contrasts, for Studies 2 and 3 (compliant runs).

For each pair of agents a, b and each study's two models (lo -> hi), the contrast is
    (mean[a, hi] - mean[a, lo]) - (mean[b, hi] - mean[b, lo])
the difference between two agents' changes when the model changes. It is reported with a normal-approximation SE
(square root of the four cells' squared SEs, the method behind the paper's other SEs) and a 95% bootstrap interval
that resamples runs within each of the four cells (20,000 draws, seed 0).

All three pairs are printed for both studies, because picking the one that clears zero after seeing all three is a
multiple comparison. Study 2's model change crosses model families and its pattern was noticed after the data were
in, so it is exploratory. Study 3's model change was planned, but which agent pair to compare was not. Both studies
start from the same GLM-5.3 Flash runs, so their contrasts are not independent; the per-model "a minus b" lines show
the parts that are.

Usage (repo root): python experiments/run_variance/interaction_table.py [STUDY2_CELLS STUDY3_CELLS]
"""
import csv, itertools, random, statistics as st, sys
from collections import defaultdict

S2 = sys.argv[1] if len(sys.argv) > 1 else "results/variance/cells.csv"
S3 = sys.argv[2] if len(sys.argv) > 2 else "results/variance_glm53/cells.csv"
AGENTS = ["pi", "hermes", "opencode"]
NAME = {"pi": "pi", "hermes": "Hermes", "opencode": "OpenCode", "glm-5.3-flash": "GLM-5.3 Flash",
        "deepseek-4.1-flash": "DeepSeek 4.1 Flash", "glm-5.3": "GLM-5.3"}


def load(path):
    out = defaultdict(list)
    for r in csv.DictReader(open(path)):
        if r["counted"] == "True" and r["status"] == "scored" and r["holdout_auc"] \
                and r["compliant"] == "True":
            out[(r["harness"], r["model"])].append(float(r["holdout_auc"]))
    return out


data = load(S2); data.update(load(S3))
mean = lambda k: st.mean(data[k])
se2 = lambda k: st.variance(data[k]) / len(data[k])
rng = random.Random(0)


def boot(cells, sign, n=20000):
    """95% bootstrap interval of sum(sign_i * mean(cell_i)), resampling runs within each cell."""
    vals = []
    for _ in range(n):
        vals.append(sum(s * st.mean(rng.choices(data[c], k=len(data[c]))) for c, s in zip(cells, sign)))
    vals.sort()
    return vals[int(0.025 * n)], vals[int(0.975 * n)]


for study, lo, hi in (("Study 2 (exploratory: model families differ, pattern seen after the data)", "glm-5.3-flash", "deepseek-4.1-flash"),
                      ("Study 3 (planned model change; the agent pair was not planned)", "glm-5.3-flash", "glm-5.3")):
    print(f"{study}\n  {NAME[lo]} -> {NAME[hi]}, compliant runs")
    for a in AGENTS:
        g = mean((a, hi)) - mean((a, lo)); s = (se2((a, hi)) + se2((a, lo))) ** .5
        print(f"  change for {NAME[a]:9s} {g:+.4f}  SE {s:.4f}  {g / s:5.1f} SE")
    print("  interaction contrasts (change for a minus change for b):")
    for a, b in itertools.combinations(AGENTS, 2):
        cells = [(a, hi), (a, lo), (b, hi), (b, lo)]
        d = mean(cells[0]) - mean(cells[1]) - mean(cells[2]) + mean(cells[3])
        s = sum(se2(c) for c in cells) ** .5
        lo_ci, hi_ci = boot(cells, (1, -1, -1, 1))
        print(f"    {NAME[a]:8s} vs {NAME[b]:9s} {d:+.4f}  SE {s:.4f}  {d / s:+5.1f} SE   bootstrap 95% {lo_ci:+.4f} to {hi_ci:+.4f}"
              f"   {'clears zero' if lo_ci > 0 or hi_ci < 0 else 'does not clear zero'}")
    # all three contrasts together: Holm-adjusted p-values, and one Wald test of "no interaction" (2 df), since only two of
    # the three contrasts are independent (a-c = (a-b) + (b-c))
    from scipy.stats import norm, chi2
    import numpy as np
    pairs = list(itertools.combinations(AGENTS, 2))
    est = []
    for a, b in pairs:
        cells = [(a, hi), (a, lo), (b, hi), (b, lo)]
        d = mean(cells[0]) - mean(cells[1]) - mean(cells[2]) + mean(cells[3])
        est.append((a, b, d, sum(se2(c) for c in cells) ** .5))
    raw = sorted(((2 * norm.sf(abs(d / s)), a, b) for a, b, d, s in est))
    adj, running = {}, 0.0
    for i, (pv, a, b) in enumerate(raw):
        running = max(running, min(1.0, (len(raw) - i) * pv)); adj[(a, b)] = (pv, running)
    for a, b, d, s in est:
        print(f"    Holm-adjusted p, {NAME[a]} vs {NAME[b]}: raw {adj[(a, b)][0]:.4f}, adjusted {adj[(a, b)][1]:.4f}")
    g = {h: (mean((h, hi)) - mean((h, lo)), se2((h, hi)) + se2((h, lo))) for h in AGENTS}
    ref = AGENTS[-1]
    vec = np.array([g[h][0] - g[ref][0] for h in AGENTS[:-1]])
    cov = np.array([[g[h][1] + g[ref][1] if h == k else g[ref][1] for k in AGENTS[:-1]] for h in AGENTS[:-1]])
    w = float(vec @ np.linalg.inv(cov) @ vec)
    print(f"  omnibus test of no agent x model interaction: Wald chi2 {w:.1f} on 2 df, p {chi2.sf(w, 2):.4f}")
    print("  the parts that do not share the GLM-5.3 Flash runs (a minus b on the second model):")
    for a, b in itertools.combinations(AGENTS, 2):
        d = mean((a, hi)) - mean((b, hi)); s = (se2((a, hi)) + se2((b, hi))) ** .5
        print(f"    {NAME[a]:8s} minus {NAME[b]:9s} on {NAME[hi]:18s} {d:+.4f}  ({d / s:+.1f} SE)")
    print()
