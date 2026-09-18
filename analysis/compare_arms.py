#!/usr/bin/env python3
"""Study 3: what the larger model in a family buys, measured against run-to-run noise.

Compares two arms of the run-variance experiment that differ in exactly one thing, the model:
  arm A (default results/variance_glm53/cells.csv)  GLM-5.3
  arm B (default results/variance/cells.csv)        GLM-5.3 Flash
Same three agents, same 52 seeds, same dataset, prompt, budget, box type and parallelism.

The screens are applied IDENTICALLY to both arms, which is the point of this script. An earlier interim comparison
read raw eval.json from the boxes, where the refit and frame-statistics screens have not run, and so compared
screened Flash runs against unscreened GLM-5.3 runs. That tilted the result toward GLM-5.3: on one box alone,
screening moved pi by -0.0040 because its best run took statistics from the frame it was scoring.

Reports, in order of how much weight they can carry:
  1. the model-tier main effect, pooled across agents
  2. the agent x model interaction: does the upgrade buy more in one agent than another?
  3. per-agent gains
  4. the detectability ladder: effect sizes in units of within-pairing SD, and the n each would need
  5. box and time effects, which are the two confounds this arm is exposed to

Usage: python experiments/run_variance/compare_arms.py [ARM_A_CELLS] [ARM_B_CELLS]
"""
import csv, math, random, statistics as st, sys
from collections import defaultdict

A_PATH = sys.argv[1] if len(sys.argv) > 1 else "results/variance_glm53/cells.csv"
B_PATH = sys.argv[2] if len(sys.argv) > 2 else "results/variance/cells.csv"
B_MODEL = "glm-5.3-flash"          # arm B holds two models; keep only the one that shares a family with arm A
BOOT = 20000
random.seed(0)


def load(path, model=None):
    """Counted, scored, rule-following runs. `counted` excludes set-aside originals of re-run cells."""
    out = []
    for r in csv.DictReader(open(path)):
        if r["counted"] != "True" or r["status"] != "scored" or not r["holdout_auc"]:
            continue
        if model and r["model"] != model:
            continue
        if r["refit_suspect"] == "True" or r["frame_stats"] == "True":
            continue                      # the same two screens on both arms, or the comparison is not like for like
        out.append(r)
    return out


def by_agent(rows):
    d = defaultdict(list)
    for r in rows:
        d[r["harness"]].append(float(r["holdout_auc"]))
    return d


def boot_ci(f, *pops):
    """Percentile CI for a statistic of one or more populations, resampling each independently."""
    vals = sorted(f(*[random.choices(p, k=len(p)) for p in pops]) for _ in range(BOOT))
    return vals[int(0.025 * BOOT)], vals[int(0.975 * BOOT)], st.stdev(vals)


def main():
    a_rows, b_rows = load(A_PATH), load(B_PATH, B_MODEL)
    A, B = by_agent(a_rows), by_agent(b_rows)
    agents = sorted(set(A) & set(B))
    if not agents:
        print("no shared agents between the arms"); return 1
    print(f"arm A (pro)   {A_PATH}: {len(a_rows)} valid runs")
    print(f"arm B (flash) {B_PATH} [{B_MODEL}]: {len(b_rows)} valid runs")
    print(f"screens applied to both: refit_suspect and frame_stats excluded\n")

    print("PER-AGENT GAIN")
    print(f"{'agent':10}{'pro':>9}{'n':>4}{'flash':>9}{'n':>4}{'gain':>9}{'95% CI':>24}")
    for h in agents:
        lo, hi, _ = boot_ci(lambda x, y: st.mean(x) - st.mean(y), A[h], B[h])
        print(f"{h:10}{st.mean(A[h]):>9.4f}{len(A[h]):>4}{st.mean(B[h]):>9.4f}{len(B[h]):>4}"
              f"{st.mean(A[h]) - st.mean(B[h]):>+9.4f}   {lo:+.4f} to {hi:+.4f}")

    ap = [v for h in agents for v in A[h]]
    bp = [v for h in agents for v in B[h]]
    gain = st.mean(ap) - st.mean(bp)
    lo, hi, se = boot_ci(lambda x, y: st.mean(x) - st.mean(y), ap, bp)
    sd = st.median([st.stdev(B[h]) for h in agents])
    print(f"\n1. MAIN EFFECT (model tier, pooled): {gain:+.4f}   95% CI {lo:+.4f} to {hi:+.4f}"
          f"   {gain/se:.1f} SE from zero")
    print(f"   within-pairing SD (flash, median across agents) {sd:.4f}  ->  the gain is {gain/sd:.2f} x the noise")
    # the same fact in plain terms: one run on each model, compared; how often does the smaller model come out ahead?
    beat = {h: sum(y > x for y in B[h] for x in A[h]) / (len(A[h]) * len(B[h])) for h in agents}
    print(f"   one run of each model: the smaller model comes out ahead {st.mean(beat.values()):.0%} of the time, "
          f"averaged over agents (" + ", ".join(f"{h} {beat[h]:.0%}" for h in agents) + ")")

    print("\n2. INTERACTION (does the upgrade buy more in one agent than another?)")
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            h1, h2 = agents[i], agents[j]
            f = lambda w, x, y, z: (st.mean(w) - st.mean(x)) - (st.mean(y) - st.mean(z))
            obs = f(A[h1], B[h1], A[h2], B[h2])
            lo2, hi2, se2 = boot_ci(f, A[h1], B[h1], A[h2], B[h2])
            verdict = "clears zero" if lo2 > 0 or hi2 < 0 else "DOES NOT clear zero"
            print(f"   {h1:9} vs {h2:9} {obs:>+9.4f}   95% CI {lo2:+.4f} to {hi2:+.4f}"
                  f"   {abs(obs)/se2:.1f} SE   {verdict}")

    print("\n3. DETECTABILITY LADDER (n per arm for 80% power at the 5% level, 16 sd^2 / gap^2)")
    ladder = [("model tier, pooled", abs(gain))]
    ma = {h: st.mean(A[h]) for h in agents}
    mb = {h: st.mean(B[h]) for h in agents}
    ladder.append(("widest agent gap, pro model", max(ma.values()) - min(ma.values())))
    ladder.append(("widest agent gap, flash model", max(mb.values()) - min(mb.values())))
    print(f"   {'effect':36}{'size':>9}{'in SDs':>9}{'n needed':>10}")
    for label, g in ladder:
        n = 16 * sd ** 2 / g ** 2 if g else float("inf")
        print(f"   {label:36}{g:>+9.4f}{g/sd:>9.2f}{n:>10.0f}")

    print("\n4. CONFOUND CHECKS on arm A")
    boxes = defaultdict(list)
    for r in a_rows:
        boxes[r.get("box", "?")].append(float(r["holdout_auc"]))
    if len(boxes) > 1:
        grand = st.mean([v for g in boxes.values() for v in g])
        k, n = len(boxes), sum(len(g) for g in boxes.values())
        between = sum(len(g) * (st.mean(g) - grand) ** 2 for g in boxes.values()) / (k - 1)
        within = sum((v - st.mean(g)) ** 2 for g in boxes.values() for v in g) / (n - k)
        print(f"   box effect: F = {between/within:.2f} across {k} boxes "
              f"({'no box effect worth reporting' if between/within < 3 else 'INVESTIGATE'})")
    order = [r for r in a_rows if r.get("finished_at") or r.get("started")]
    if len(order) > 20:
        key = "finished_at" if a_rows[0].get("finished_at") else "started"
        order.sort(key=lambda r: r[key])
        half = len(order) // 2
        e = [float(r["holdout_auc"]) for r in order[:half]]
        l = [float(r["holdout_auc"]) for r in order[half:]]
        print(f"   time effect: early half {st.mean(e):.4f}, late half {st.mean(l):.4f}, drift {st.mean(l)-st.mean(e):+.4f}")
    else:
        print("   time effect: no timestamp column in cells.csv; check separately")
    return 0


if __name__ == "__main__":
    sys.exit(main())
