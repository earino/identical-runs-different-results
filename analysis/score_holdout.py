"""Score a train.py against a labeled CSV. Run from inside a workdir copy:
    python -m bench.score_holdout <csv> <target> <positive_label_json>
Prints one JSON line at the end: {"auc":..., "ap":..., "n":..., "seconds":...}
"""
from __future__ import annotations

import json
import runpy
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def main() -> None:
    csv, target, pos = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
    t0 = time.time()
    ns = runpy.run_path("train.py", run_name="__train__")
    fn = ns.get("predict_proba")
    if not callable(fn):
        print(json.dumps({"error": "no predict_proba"}))
        return
    df = pd.read_csv(csv)
    y = (df[target] == pos).astype(int).to_numpy()
    p = np.asarray(fn(df.drop(columns=[target])), dtype=float).ravel()
    if p.shape[0] != len(df) or not np.all(np.isfinite(p)):
        print(json.dumps({"error": f"bad predictions shape={p.shape} finite={bool(np.all(np.isfinite(p)))}"}))
        return
    print(json.dumps({
        "auc": float(roc_auc_score(y, p)),
        "ap": float(average_precision_score(y, p)),
        "n": int(len(df)),
        "positives": int(y.sum()),
        "seconds": round(time.time() - t0, 1),
    }))


if __name__ == "__main__":
    main()
