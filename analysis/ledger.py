#!/usr/bin/env python3
"""The run ledger: every count the paper quotes, from the per-run tables, with assertions that they reconcile.

One place for flow counts, exclusions and compliance yields, so text, tables and figures cannot drift apart. Each
assertion states an identity the paper relies on; if a table changes and a count no longer reconciles, this fails.

Study 1  results/phase2/merged/cells.csv   (six complete model rows plus a partial V4 Pro row)
Study 2  results/variance/cells.csv        (compliant = no evaluation labels in the delivered code's training and no
Study 3  results/variance_glm53/cells.csv   statistics from the scored frame; see audit_eval_training.py)

Usage (repo root): python experiments/run_variance/ledger.py [S1 S2 S3]
"""
import csv, statistics as st, sys
from collections import Counter, defaultdict

S1 = sys.argv[1] if len(sys.argv) > 1 else "results/phase2/merged/cells.csv"
S2 = sys.argv[2] if len(sys.argv) > 2 else "results/variance/cells.csv"
S3 = sys.argv[3] if len(sys.argv) > 3 else "results/variance_glm53/cells.csv"
T = lambda v: v == "True"

# ---- Study 1
s1 = list(csv.DictReader(open(S1)))
partial = [r for r in s1 if r["model"].startswith("deepseek-v4-pro")]
complete = [r for r in s1 if not r["model"].startswith("deepseek-v4-pro")]
never = [r for r in complete if (r["n_experiments"] or "0") in ("0", "0.0")]
noncompliant1 = [r for r in complete if T(r["refit_suspect"]) and r not in never]
scored1 = [r for r in complete if r not in never and r not in noncompliant1]
print("STUDY 1")
print(f"  runs recorded {len(s1)} = {len(complete)} on the six complete rows + {len(partial)} on the partial V4 Pro row")
print(f"  never started {len(never)} ({', '.join(sorted({r['harness'] + ' / ' + r['model'] for r in never}))}); "
      f"their files are the untouched starting code")
print(f"  delivered by an agent {len(s1) - len(never)}; complete-row artifacts {len(complete) - len(never)}, "
      f"noncompliant {len(noncompliant1)}, scored in the quality analysis {len(scored1)}")
assert len(complete) == 108 and len(partial) == 8 and len(s1) == 116
assert len(complete) - len(never) - len(noncompliant1) == len(scored1)


def study(path, name):
    rows = [r for r in csv.DictReader(open(path)) if T(r["counted"])]
    st_ = Counter(r["status"] for r in rows)
    scored = [r for r in rows if r["status"] == "scored"]
    ev = [r for r in scored if T(r["eval_trained"])]; fr = [r for r in scored if T(r["frame_stats"])]
    both = [r for r in ev if T(r["frame_stats"])]
    comp = [r for r in scored if T(r["compliant"])]
    print(name)
    print(f"  runs counted {len(rows)}; status {dict(st_)}")
    print(f"  trained on evaluation labels {len(ev)}; statistics from the scored frame {len(fr)}; both {len(both)}; "
          f"broke a rule {len(ev) + len(fr) - len(both)}")
    print(f"  compliant {len(comp)} of {len(rows)} ({len(comp) / len(rows):.1%}); excluded {len(rows) - len(comp)} = "
          f"{len(rows) - len(scored)} not scored + {len(ev) + len(fr) - len(both)} broke a rule")
    # identities the paper relies on
    assert len(comp) == len(scored) - (len(ev) + len(fr) - len(both))
    assert all(not T(r["compliant"]) for r in rows if r["status"] != "scored")
    screen_only = [r for r in scored if T(r["refit_suspect"]) and not T(r["eval_trained"])]
    screen_missed = [r for r in ev if not T(r["refit_suspect"])]
    print(f"  the old score screen: flagged {sum(T(r['refit_suspect']) for r in scored)}, of which "
          f"{len(screen_only)} did not train on evaluation labels "
          f"({', '.join(r['harness'] + ' seed' + r['seed'] + (' (batch features)' if T(r['frame_stats']) else '') for r in screen_only)}); "
          f"missed {len(screen_missed)} ({', '.join(r['harness'] + ' seed' + r['seed'] for r in screen_missed) or 'none'})")
    per = defaultdict(lambda: [0, 0])
    for r in rows:
        per[(r["harness"], r["model"])][0] += 1
        per[(r["harness"], r["model"])][1] += T(r["compliant"])
    for k in sorted(per):
        n, c = per[k]
        print(f"    {k[0]:9s} {k[1]:19s} compliant {c:3d} of {n}   noncompliant or unscored {n - c} ({(n - c) / n:.0%})")
    return rows, comp


s2rows, s2comp = study(S2, "STUDY 2")
assert len(s2rows) == 312
top = sorted([r for r in s2rows if r["status"] == "scored"], key=lambda r: -float(r["holdout_auc"]))[:10]
print(f"  of the ten highest Study 2 scores, {sum(not T(r['compliant']) for r in top)} broke a rule")
ev2 = sorted(float(r["holdout_auc"]) for r in s2rows if T(r["eval_trained"]))
fr2 = sorted(float(r["holdout_auc"]) for r in s2rows if T(r["frame_stats"]))
print(f"  score ranges: evaluation-label training {ev2[0]:.4f} to {ev2[-1]:.4f}; batch features {fr2[0]:.4f} to {fr2[-1]:.4f}")

s3rows, s3comp = study(S3, "STUDY 3")
assert len(s3rows) == 156

# ---- Study 3 against its comparison arm, which is Study 2's GLM-5.3 Flash runs
flash = [r for r in s2rows if r["model"] == "glm-5.3-flash"]
fc = [r for r in flash if T(r["compliant"])]
print("STUDY 3 ARMS")
print(f"  GLM-5.3 Flash (from Study 2): compliant {len(fc)} of {len(flash)} ({len(fc) / len(flash):.1%}); "
      f"GLM-5.3: compliant {len(s3comp)} of {len(s3rows)} ({len(s3comp) / len(s3rows):.1%})")
for name, rs in (("GLM-5.3 Flash", flash), ("GLM-5.3", s3rows)):
    started = sorted(r["started"] for r in rs if r["started"]); boxes = sorted({r["box"] for r in rs})
    print(f"  {name:14s} runs {started[0][:16]} to {started[-1][:16]} UTC on machines {', '.join(boxes)}")
by = lambda rs: {h: [float(r["holdout_auc"]) for r in rs if r["harness"] == h] for h in ("pi", "hermes", "opencode")}
fa, ga = by(fc), by(s3comp)
pool = lambda d: st.mean([x for v in d.values() for x in v]); eq = lambda d: st.mean(st.mean(v) for v in d.values())
print(f"  means, pooled over compliant runs: Flash {pool(fa):.4f}, GLM-5.3 {pool(ga):.4f}, gain {pool(ga) - pool(fa):+.4f}")
print(f"  means, agents weighted equally:    Flash {eq(fa):.4f}, GLM-5.3 {eq(ga):.4f}, gain {eq(ga) - eq(fa):+.4f}")
for h in ("pi", "hermes", "opencode"):
    print(f"    {h:9s} GLM-5.3 {st.mean(ga[h]):.4f} ({len(ga[h])} runs)   Flash {st.mean(fa[h]):.4f} ({len(fa[h])} runs)   "
          f"gain {st.mean(ga[h]) - st.mean(fa[h]):+.4f}")
print("\nall ledger identities hold")
