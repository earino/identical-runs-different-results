#!/usr/bin/env python3
"""Best-of-k: what attempting a job k times and keeping the best compliant result buys (the paper's Study 2 table).

The policy: draw k attempts from a pairing's observed runs, with replacement; reject noncompliant attempts; keep the
one whose delivered code scores best on the evaluation set; report the kept code's holdout AUC. The six pairings are
weighted equally. Everything is exact rather than simulated: with replacement, the attempt with the r-th lowest
selection score of a pairing's n runs is kept with probability (r/n)^k - ((r-1)/n)^k, and no compliant attempt is
drawn with probability f^k, f being the pairing's share of noncompliant runs.

The evaluation score is that of the code the run delivered (its final train.py, which in 18 of the 312 runs was not
its best experiment; export_final_eval.py extracts it). For comparison the script also chooses on holdout AUC itself,
an oracle no user has; the difference in the mean kept score is the winner's-curse optimism of choosing on the set
that scores. It is small here because a score on the 1M-row holdout has a standard error of about 0.0005, twenty
times smaller than the spread between runs.

policy_tables.py, best_of_k_by_pairing.py and sensitivity_violations.py import load, pooled and quantile from here.

Usage (repo root): python experiments/run_variance/best_of_k.py [CELLS_CSV [FINAL_EVAL_CSV]]
  FINAL_EVAL_CSV comes from export_final_eval.py (default results/variance/final_eval.csv)
"""
import csv, sys
from collections import defaultdict

KS = (1, 3, 5, 10)


def load(cells, final, compliant=lambda r: r["compliant"] == "True"):
    """Every counted run of each pairing as (holdout AUC, evaluation AUC of the delivered code, compliant). Runs that
    could not be scored count as noncompliant attempts. One Study 3 run ran no experiments and delivered the starting
    code; it has no evaluation score, so it carries 0 and is kept only when no other compliant attempt is drawn."""
    fe = {(r["harness"], r["model"], r["seed"]): r["final_eval_auc"] for r in csv.DictReader(open(final))}
    pairs = defaultdict(list)
    for r in csv.DictReader(open(cells)):
        if r["counted"] != "True":
            continue
        ok = r["status"] == "scored" and compliant(r)
        e = fe.get((r["harness"], r["model"], r["seed"]))
        pairs[(r["harness"], r["model"])].append((float(r["holdout_auc"]) if ok else None,
                                                  float(e) if e not in (None, "", "None") else 0.0, ok))
    return pairs


def kept(runs, k, by="eval"):
    """Exact distribution of the kept attempt in one pairing: (chance no attempt is compliant, [(holdout, prob)]).
    by="eval" chooses on the evaluation score, by="holdout" on the holdout score. Tied scores share their probability
    equally, which is what keeping the first of the tied attempts drawn does."""
    n = len(runs)
    below = sum(not ok for *_, ok in runs)                    # noncompliant attempts rank below every compliant one
    comp = sorted(((h if by == "holdout" else e), h) for h, e, ok in runs if ok)
    out, i = [], 0
    while i < len(comp):
        j = i
        while j < len(comp) and comp[j][0] == comp[i][0]:
            j += 1
        p = ((below + j - i) / n) ** k - (below / n) ** k
        out += [(h, p / (j - i)) for _, h in comp[i:j]]
        below += j - i; i = j
    return (sum(not ok for *_, ok in runs) / n) ** k, out


def pooled(pairs, k, by="eval"):
    """The pairings weighted equally: (chance of at least one compliant attempt, [(holdout, prob)] sorted)."""
    none, mass = 0.0, []
    for runs in pairs.values():
        f, dist = kept(runs, k, by)
        none += f / len(pairs); mass += [(h, p / len(pairs)) for h, p in dist]
    return 1 - none, sorted(mass)


def quantile(mass, q):
    """The q-quantile of the kept score, given that an artifact was delivered."""
    total, c = sum(p for _, p in mass), 0.0
    for h, p in mass:
        c += p
        if c >= q * total - 1e-12:
            return h
    return mass[-1][0]


def mean(mass):
    return sum(h * p for h, p in mass) / sum(p for _, p in mass)


if __name__ == "__main__":
    CELLS = sys.argv[1] if len(sys.argv) > 1 else "results/variance/cells.csv"
    FINAL = sys.argv[2] if len(sys.argv) > 2 else "results/variance/final_eval.csv"
    pairs = load(CELLS, FINAL)
    print(f"best-of-k over {len(pairs)} pairings weighted equally, exact; kept attempt's holdout AUC\n")
    print(f"{'k':>3}  {'>=1 compliant':>13}   {'chosen on eval: median  p5      p95':36}   "
          f"{'oracle, chosen on holdout: median  p5      p95':47}   optimism")
    q = {}
    for k in KS:
        ok, me = pooled(pairs, k, "eval"); _, mh = pooled(pairs, k, "holdout")
        qe = [round(quantile(me, x), 4) for x in (.5, .05, .95)]; qh = [quantile(mh, x) for x in (.5, .05, .95)]
        q[k] = qe
        print(f"{k:>3}  {ok:>13.4%}   {'':16}{qe[0]:.4f}  {qe[1]:.4f}  {qe[2]:.4f}   {'':27}{qh[0]:.4f}  {qh[1]:.4f}  "
              f"{qh[2]:.4f}   {mean(mh) - mean(me):+.5f}")
    # differences of the table's own rounded values, so text and table agree
    print(f"\nmedian kept: three attempts {q[3][0] - q[1][0]:+.4f} over one, ten attempts {q[10][0] - q[1][0]:+.4f}")
    print(f"one attempt to ten: 5th percentile {q[10][1] - q[1][1]:+.4f}, 95th percentile {q[10][2] - q[1][2]:+.4f}")
