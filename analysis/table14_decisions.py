#!/usr/bin/env python3
"""Table 14: what an AUC gap is worth at one operating point, from two delivered models' real predictions.

Takes pi's best compliant run on DeepSeek 4.1 Flash (seed 41, holdout AUC 0.7695) and a run near that pairing's
median (seed 16, 0.7471; the pairing's compliant median is 0.7462), retrains each delivered program exactly as
rescore.py does, and scores the 1,000,000-row holdout. The holdout has the classes in equal numbers, so each model's
true- and false-positive rates at every score threshold are carried to a team that sees 1,000,000 cases a month, one
in a hundred positive (only the prevalence differs from the benchmark). The team reviews the N highest-scoring cases;
a missed positive costs $100 and a false alarm $10. Difference = monthly cost with the 0.7471 model minus with the
0.7695 model: positive means the higher-AUC model saves money. Illustrative, not a forecast: both models were
selected and scored on the same holdout.

Usage (repository root; a few minutes per model):  python analysis/table14_decisions.py
"""
import os, shutil, subprocess, sys, tempfile
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
RUNS = {"0.7695": "code/study2/var4/airline/pi/deepseek-4.1-flash/seed41",
        "0.7471": "code/study2/var2/airline/pi/deepseek-4.1-flash/seed16"}
CASES, PREVALENCE, FALSE_ALARM, MISS = 1_000_000, 0.01, 10, 100
REVIEWED = (5_000, 20_000, 100_000)
TASK_KEYS = ("name", "description", "target", "positive_label", "id_columns", "split", "columns", "rows")


CHILD = r"""
import json, os, runpy, shutil, sys, tempfile
from pathlib import Path
import numpy as np, pandas as pd
root, cell, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
task = root / "task" / "airline"; meta = json.loads((task / "meta.json").read_text())
keys = ("name", "description", "target", "positive_label", "id_columns", "split", "columns", "rows")
work = Path(tempfile.mkdtemp(prefix="table14-"))
try:
    for f in (root / "harness" / "task_template").iterdir():
        if f.is_file() and f.name not in ("program.md", "research_section.md", "no_research_section.md"):
            shutil.copy2(f, work / f.name)
    (work / "task.json").write_text(json.dumps({k: meta[k] for k in keys if k in meta}))
    (work / "data").mkdir()
    for name in ("train.csv", "eval.csv"):
        shutil.copy2(task / name, work / "data" / name)
    shutil.copy2(sorted((root / cell).glob("code/*.py"))[-1], work / "train.py")
    os.chdir(work)
    fn = runpy.run_path("train.py", run_name="__train__")["predict_proba"]
    df = pd.read_csv(task / "holdout.csv")
    y = (df[meta["target"]] == meta["positive_label"]).astype(int).to_numpy()
    np.savez(out, y=y, p=np.asarray(fn(df.drop(columns=[meta["target"]])), dtype=float).ravel())
finally:
    shutil.rmtree(work, ignore_errors=True)
"""


def predictions(cell):
    """Holdout labels and scores of the last delivered train.py of one run. Each program trains in its own process
    with the thread settings of score_2007.py, as the benchmark's scorer did: run in one shared process, the second
    program inherits the first one's imports and random state and does not reproduce its recorded score."""
    out = Path(tempfile.mkdtemp(prefix="table14-")) / "pred.npz"
    env = dict(os.environ, OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    subprocess.run([sys.executable, "-c", CHILD, str(ROOT), cell, str(out)], check=True, env=env,
                   stdout=subprocess.DEVNULL)
    d = np.load(out); shutil.rmtree(out.parent, ignore_errors=True)
    return d["y"], d["p"]


def at_capacity(y, p, reviewed):
    """(positives caught, monthly cost) when the top `reviewed` of CASES are checked, at PREVALENCE."""
    order = np.argsort(-p, kind="mergesort"); ys = y[order]
    tpr = np.cumsum(ys) / ys.sum(); fpr = np.cumsum(1 - ys) / (len(ys) - ys.sum())
    pos, neg = CASES * PREVALENCE, CASES * (1 - PREVALENCE)
    load = pos * tpr + neg * fpr                        # cases reviewed at each threshold, increasing
    caught = float(np.interp(reviewed, load, pos * tpr))
    return caught, MISS * (pos - caught) + FALSE_ALARM * (reviewed - caught)


if __name__ == "__main__":
    res = {}
    for label, cell in RUNS.items():
        y, p = predictions(cell)
        res[label] = (y, p)
        print(f"{cell}: holdout AUC {roc_auc_score(y, p):.4f}")
    print(f"\n{CASES} cases a month, prevalence {PREVALENCE:.0%}, false alarm ${FALSE_ALARM}, miss ${MISS}")
    print(f"{'reviewed':>9}  {'caught 0.7695':>13}  {'caught 0.7471':>13}  {'difference':>10}")
    for n in REVIEWED:
        cb, kb = at_capacity(*res["0.7695"], n); cm, km = at_capacity(*res["0.7471"], n)
        print(f"{n:>9}  {cb:>13.0f}  {cm:>13.0f}  {km - kb:>10.0f}")
