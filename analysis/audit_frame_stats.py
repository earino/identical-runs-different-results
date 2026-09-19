#!/usr/bin/env python3
"""Find runs whose delivered train.py computes statistics FROM THE FRAME BEING SCORED.

program.md, contract rule 2: "Fit encoders/statistics on training data only; never on the dataframe passed in."
A run that breaks it reads the holdout's own distribution (counts, shares, group means) when predict_proba is called on
the holdout. Neither existing check catches it: the violation scan greps for filesystem/network idioms, and the refit
screen only fires when eval AUC far exceeds holdout, while frame-derived features raise BOTH (hermes/deepseek-4.1-flash
seed15: eval 0.7965, holdout 0.8036, +0.058 over its pair's mean, from `X[c].value_counts()/n` inside prepare()).

How it decides (AST, not grep): start at predict_proba, follow calls to functions defined in the file, and taint each
function's own parameters plus everything assigned from them. A statistic called on a tainted value is a hit; the same
call on a module-level table fitted from `train` is not. `len(df)` is ignored (allocating an output array is fine).

This is a screen, not a verdict: read every hit (same rule as scripts/phase2/audit_eval_leak.py).

Usage (from the main checkout):
  python <worktree>/experiments/run_variance/audit_frame_stats.py [PULLS] [OUT_CSV]
"""
import ast, csv, glob, json, os, statistics as st, subprocess, sys
from collections import Counter, deque

PULLS = sys.argv[1] if len(sys.argv) > 1 else "pulls/variance"
OUT = sys.argv[2] if len(sys.argv) > 2 else "results/variance/frame_stats.csv"
# Frame-wide statistics only. Row-wise work (.agg(axis=1)) and averaging ensemble predictions (.mean(axis=0)) say
# nothing about the frame's distribution, so they are not listed: they produced only false positives.
STATS = {"value_counts", "nunique", "rank", "groupby", "transform"}


def base_name(node):
    """Root identifier of an expression: X[c].map(...) -> X ; df.groupby(a)[b] -> df."""
    while True:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, (ast.Subscript, ast.Attribute)):
            node = node.value
        elif isinstance(node, ast.Call):
            node = node.func
        else:
            return None


def audit(src):
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return [], "unparsable"
    funcs = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    if "predict_proba" not in funcs:
        return [], "no predict_proba"
    # Interprocedural taint: only predict_proba's own argument is the scored frame. A callee's parameter becomes
    # tainted only when a tainted value is passed into that position, so a helper that a predict call also invokes
    # with the module-level training frame (e.g. building OOF encodings) is not flagged.
    def local_taint(fn, seeded):
        tainted = set(seeded)
        for _ in range(3):                                             # propagate through assignments
            for node in ast.walk(fn):
                if isinstance(node, ast.Assign):
                    if any(base_name(v) in tainted for v in ast.walk(node.value) if isinstance(v, (ast.Name, ast.Subscript, ast.Attribute))):
                        for t in node.targets:
                            if base_name(t):
                                tainted.add(base_name(t))
        return tainted

    params = lambda fn: [a.arg for a in fn.args.args]
    start = params(funcs["predict_proba"])[:1]                          # predict_proba(df): df is the holdout
    work, seen, hits = deque([("predict_proba", tuple(start))]), set(), []
    while work:
        fname, seeded = work.popleft()
        if (fname, seeded) in seen or fname not in funcs:
            continue
        seen.add((fname, seeded))
        fn = funcs[fname]
        tainted = local_taint(fn, seeded)
        for node in ast.walk(fn):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in STATS \
                    and base_name(node.func.value) in tainted:
                hits.append((node.lineno, node.func.attr, src.splitlines()[node.lineno - 1].strip()[:120], fname))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in funcs:
                callee = funcs[node.func.id]
                names = params(callee)
                passed = tuple(names[i] for i, a in enumerate(node.args)
                               if i < len(names) and base_name(a) in tainted)
                if passed:
                    work.append((node.func.id, passed))
    return hits, ""


def audit_run(run_dir):
    """(hits, note) for one run dir, or ([], 'no train.py'). Used by analyze.py to mark every run."""
    # the scored file is the committed HEAD (the scorer runs `git archive HEAD`), not the working copy
    wd = os.path.join(run_dir, "workdir")
    head = subprocess.run(["git", "-C", wd, "show", "HEAD:train.py"], capture_output=True, text=True)
    if head.returncode == 0:
        return audit(head.stdout)
    tp = os.path.join(wd, "train.py")
    if not os.path.exists(tp):
        return [], "no train.py"
    return audit(open(tp, errors="replace").read())


def main():
    rows = []
    for run in sorted(glob.glob(os.path.join(PULLS, "*", "runs", "airline", "*", "*", "seed*"))):
        hits, note = audit_run(run)
        if note == "no train.py":
            continue
        parts = run.rstrip("/").split(os.sep)
        box, harness, model, seed = parts[-5], parts[-3], parts[-2], parts[-1]
        e = json.load(open(os.path.join(run, "eval.json"))) if os.path.exists(os.path.join(run, "eval.json")) else {}
        rows.append({"box": box, "harness": harness, "model": model, "seed": int(seed.replace("seed", "")),
                     "holdout_auc": e.get("holdout_auc"), "best_eval_auc": e.get("best_eval_auc"),
                     "frame_stats": bool(hits), "n_hits": len(hits), "note": note,
                     "kinds": ";".join(sorted({h[1] for h in hits})),
                     "where": ";".join(sorted({h[3] for h in hits})),
                     "lines": " | ".join(f"{h[0]}: {h[2]}" for h in hits[:5])})
    if not rows:
        sys.exit(f"no train.py under {PULLS}/*/runs/airline/*/*/seed*")
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    flag = [r for r in rows if r["frame_stats"] and r["holdout_auc"]]
    rest = [r for r in rows if not r["frame_stats"] and r["holdout_auc"]]
    print(f"{len(rows)} runs scanned -> {OUT}")
    print(f"  statistics computed on the scored frame: {len(flag)} runs ({100 * len(flag) / len(rows):.0f}%)")
    if flag and rest:
        print(f"  mean holdout   flagged {st.mean([r['holdout_auc'] for r in flag]):.4f} (n={len(flag)})   others {st.mean([r['holdout_auc'] for r in rest]):.4f} (n={len(rest)})")
    print("  by pair:", dict(Counter(f"{r['harness']}/{r['model']}" for r in flag)))
    if any(r["note"] for r in rows):
        print("  notes:", dict(Counter(r["note"] for r in rows if r["note"])))
    print("\n  READ EVERY HIT (screen, not verdict). Highest scoring flagged runs:")
    for r in sorted(flag, key=lambda r: -r["holdout_auc"])[:8]:
        print(f"   {r['harness']}/{r['model']} seed{r['seed']} holdout={r['holdout_auc']:.4f} in {r['where']} [{r['kinds']}]\n     {r['lines'][:220]}")


if __name__ == "__main__":
    main()
