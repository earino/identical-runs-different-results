#!/usr/bin/env python3
"""Score every delivered program of Studies 2 and 3 on 2007 flights, which no part of this study had used.

Each run's delivered train.py (the last version in the exported code, which is the scored file) is retrained from
scratch in a fresh working directory, exactly as the holdout scorer does, and its predict_proba is applied to two
sets in the same process: the 2006 holdout the paper scores on, and a balanced 1,000,000-row slice of 2007 flights
built by the same preparation (datasets/prepare_airline.py) and never shown to an agent or used in any analysis
before. Scoring both from one fit makes the 2006 column a check: it should reproduce each run's recorded holdout AUC
to within the retraining noise, and the 2007 column is then a second, later test set for the same artifacts.

Output: one row per run with auc_2006_rescored, auc_2007 and seconds. Runs are scored four at a time, four threads
each; the script resumes, skipping runs already in the output.

Usage (repo root):
  python experiments/run_variance/score_2007.py [OUT_CSV]      (benchmark repository)
  python analysis/score_2007.py [OUT_CSV]                      (data repository: task/airline/flights_2007.csv)
Scoring all 464 runs takes about six hours on 16 cores.
"""
import csv, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / "task/airline/holdout.csv").exists():            # the data repository's layout
    OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "data/2007/scores.csv")
    STUDIES = {"study2": (ROOT / "data/study2/cells.csv", ROOT / "code/study2"),
               "study3": (ROOT / "data/study3/cells.csv", ROOT / "code/study3")}
    HOLDOUT_2006, SET_2007 = ROOT / "task/airline/holdout.csv", ROOT / "task/airline/flights_2007.csv"
    META, TEMPLATE, TRAIN_EVAL = ROOT / "task/airline/meta.json", ROOT / "harness/task_template", ROOT / "task/airline"
else:                                                      # the benchmark repository's layout
    ROOT = ROOT.parent
    OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "results/variance_2007/scores.csv")
    STUDIES = {"study2": (ROOT / "results/variance/cells.csv", ROOT / "results/variance/evals"),
               "study3": (ROOT / "results/variance_glm53/cells.csv", ROOT / "results/variance_glm53/evals")}
    HOLDOUT_2006, SET_2007 = ROOT / "data/prepared/airline/private/holdout.csv", ROOT / "data/raw/airline/2007-slice2-1m.csv"
    META, TEMPLATE, TRAIN_EVAL = ROOT / "data/prepared/airline/meta.json", ROOT / "task_template", ROOT / "data/prepared/airline/public"
WORKERS, THREADS, TIMEOUT = 4, 4, 3600

# The child process: build the working directory the agent had, import train.py (which trains), then score both sets.
CHILD = r'''
import json, os, runpy, shutil, sys, tempfile, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
code, template, train_eval, meta_path, sets = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4], sys.argv[5:]
meta = json.load(open(meta_path))
keys = ("name", "description", "target", "positive_label", "id_columns", "split", "columns", "rows")
work = Path(tempfile.mkdtemp(prefix="score2007-"))
try:
    for f in template.iterdir():
        if f.is_file() and f.name not in ("program.md", "research_section.md", "no_research_section.md"):
            shutil.copy2(f, work / f.name)
    (work / "task.json").write_text(json.dumps({k: meta[k] for k in keys if k in meta}))
    (work / "data").mkdir()
    for name in ("train.csv", "eval.csv"):
        shutil.copy2(train_eval / name, work / "data" / name)
    shutil.copy2(code, work / "train.py")
    t0 = time.time(); os.chdir(work)
    ns = runpy.run_path("train.py", run_name="__train__")
    out = {"seconds_fit": round(time.time() - t0, 1)}
    for tag, path in zip(("auc_2006_rescored", "auc_2007"), sets):
        df = pd.read_csv(path)
        y = (df[meta["target"]] == meta["positive_label"]).astype(int).to_numpy()
        p = np.asarray(ns["predict_proba"](df.drop(columns=[meta["target"]])), dtype=float).ravel()
        out[tag] = float(roc_auc_score(y, p))
    print("RESULT " + json.dumps(out))
finally:
    shutil.rmtree(work, ignore_errors=True)
'''

FIELDS = ["study", "box", "harness", "model", "seed", "compliant", "holdout_auc", "auc_2006_rescored", "auc_2007",
          "seconds", "error"]


def runs():
    for study, (cells, evals) in STUDIES.items():
        for r in csv.DictReader(open(cells)):
            if r["counted"] != "True" or r["status"] != "scored":
                continue
            code = sorted((evals / r["box"] / "airline" / r["harness"] / r["model"] / f"seed{r['seed']}" / "code").glob("*.py"))
            yield study, r, (code[-1] if code else None)


def score(job):
    study, r, code = job
    row = {"study": study, "box": r["box"], "harness": r["harness"], "model": r["model"], "seed": r["seed"],
           "compliant": r["compliant"], "holdout_auc": r["holdout_auc"], "auc_2006_rescored": "", "auc_2007": "",
           "seconds": "", "error": ""}
    if code is None:
        row["error"] = "no exported code"; return row
    env = dict(os.environ, OMP_NUM_THREADS=str(THREADS), OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    t0 = time.time()
    try:
        p = subprocess.run([sys.executable, "-c", CHILD, str(code), str(TEMPLATE), str(TRAIN_EVAL), str(META),
                            str(HOLDOUT_2006), str(SET_2007)], capture_output=True, text=True, timeout=TIMEOUT, env=env)
        line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
        if line:
            res = json.loads(line[-1][7:])
            row.update(auc_2006_rescored=f"{res['auc_2006_rescored']:.6f}", auc_2007=f"{res['auc_2007']:.6f}")
        else:
            row["error"] = (p.stderr.strip().splitlines() or ["no result"])[-1][:200]
    except subprocess.TimeoutExpired:
        row["error"] = f"timeout after {TIMEOUT}s"
    row["seconds"] = f"{time.time() - t0:.0f}"
    return row


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if OUT.exists():
        done = {(r["study"], r["harness"], r["model"], r["seed"]) for r in csv.DictReader(open(OUT)) if not r["error"]}
    jobs = [j for j in runs() if (j[0], j[1]["harness"], j[1]["model"], j[1]["seed"]) not in done]
    print(f"{len(done)} already scored, {len(jobs)} to go", flush=True)
    new = not OUT.exists()
    with open(OUT, "a", newline="") as fh, ThreadPoolExecutor(WORKERS) as pool:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for i, fut in enumerate(as_completed([pool.submit(score, j) for j in jobs]), 1):
            row = fut.result()
            w.writerow(row); fh.flush()
            print(f"{i}/{len(jobs)} {row['study']} {row['harness']} {row['model']} seed{row['seed']} "
                  f"2006 {row['auc_2006_rescored'] or '-'} (recorded {row['holdout_auc'][:8]}) 2007 {row['auc_2007'] or '-'} "
                  f"{row['seconds']}s {row['error']}", flush=True)
