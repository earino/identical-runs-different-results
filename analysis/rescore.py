#!/usr/bin/env python3
"""Re-score a delivered program against the holdout, end to end.

This is the strongest check in the repository: it takes the code an agent actually wrote, runs it on the 1,000,000
row holdout the agent never saw, and prints the AUC. If our numbers were wrong or our scoring was doing something
other than what the paper says, this is where it would show.

It rebuilds the working directory the agent had -- the task scaffold, task.json derived from the dataset meta, and
data/train.csv plus data/eval.csv -- then imports the delivered train.py and calls its predict_proba on the holdout
with the label column removed. That is exactly what the benchmark's own scorer does.

Usage:
    python analysis/rescore.py code/study3/var7/airline/pi/glm-5.3/seed35
    python analysis/rescore.py code/study3/var7/airline/pi/glm-5.3/seed35 --version 03   # an earlier commit

By default it scores the LAST version of train.py, which is the one the paper scored. Compare the printed AUC with
the holdout_auc for that run in data/study3/cells.csv (or data/study2/cells.csv for study 2).

Expect a few minutes per run: these programs fit real models, some of them 16-member ensembles.
"""
import argparse, json, runpy, shutil, sys, tempfile, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TASK_KEYS = ("name", "description", "target", "positive_label", "id_columns", "split", "columns", "rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cell", help="a directory under code/, e.g. code/study3/var7/airline/pi/glm-5.3/seed35")
    ap.add_argument("--version", help="two-digit prefix of a train.py version; default is the last one")
    ap.add_argument("--dataset", default="airline")
    args = ap.parse_args()

    cell = Path(args.cell)
    versions = sorted(cell.glob("code/*.py"))
    if not versions:
        sys.exit(f"no delivered code under {cell}/code/")
    chosen = versions[-1]
    if args.version:
        match = [v for v in versions if v.name.startswith(args.version)]
        if not match:
            sys.exit(f"no version starting {args.version}; have {[v.name for v in versions]}")
        chosen = match[0]

    task_dir = ROOT / "task" / args.dataset
    meta = json.loads((task_dir / "meta.json").read_text())
    holdout = task_dir / "holdout.csv"
    if not holdout.exists():
        sys.exit(f"{holdout} is missing: this repository ships the holdout, so it should be here")

    work = Path(tempfile.mkdtemp(prefix="rescore-"))
    try:
        for f in (ROOT / "harness" / "task_template").iterdir():
            if f.is_file() and f.name not in ("program.md", "research_section.md", "no_research_section.md"):
                shutil.copy2(f, work / f.name)
        (work / "task.json").write_text(json.dumps({k: meta[k] for k in TASK_KEYS if k in meta}, indent=2))
        (work / "data").mkdir(exist_ok=True)
        for name in ("train.csv", "eval.csv"):
            shutil.copy2(task_dir / name, work / "data" / name)
        shutil.copy2(chosen, work / "train.py")

        import numpy as np, pandas as pd
        from sklearn.metrics import average_precision_score, roc_auc_score

        cwd = Path.cwd()
        t0 = time.time()
        try:
            import os; os.chdir(work)
            ns = runpy.run_path("train.py", run_name="__train__")
        finally:
            import os; os.chdir(cwd)
        fn = ns.get("predict_proba")
        if not callable(fn):
            sys.exit("the delivered train.py defines no predict_proba")

        df = pd.read_csv(holdout)
        target, pos = meta["target"], meta["positive_label"]
        y = (df[target] == pos).astype(int).to_numpy()
        p = np.asarray(fn(df.drop(columns=[target])), dtype=float).ravel()
        print(json.dumps({"cell": str(cell), "version": chosen.name,
                          "auc": round(float(roc_auc_score(y, p)), 12),
                          "ap": round(float(average_precision_score(y, p)), 6),
                          "n": int(len(df)), "seconds": round(time.time() - t0, 1)}, indent=2))
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
