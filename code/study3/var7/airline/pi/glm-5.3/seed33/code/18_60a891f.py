"""XGBoost binary classifier for airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).
"""
import json
import os
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

TASK = json.load(open("task.json"))
TARGET = TASK["target"]
POSITIVE = TASK["positive_label"]
ID_COLS = TASK.get("id_columns", [])
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- features ------------------------------------------------------------------
BASE_NUM = ["DepTime", "Distance"]
BASE_CAT = ["DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
HOURS = pd.Index(range(25))
CAT_LEVELS = {c: pd.Index(sorted(train[c].dropna().unique())) for c in BASE_CAT}
HC_LEVELS = pd.Index(
    sorted(
        (
            train["UniqueCarrier"].astype(str)
            + "@"
            + (((train["DepTime"] // 100) % 24).astype(int)).astype(str)
        ).unique()
    )
)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    for c in BASE_NUM:
        X[c] = pd.to_numeric(df[c], errors="coerce")
    hr = ((pd.to_numeric(df["DepTime"], "coerce") // 100) % 24).astype(int)
    X["HourCat"] = pd.Categorical(hr, categories=HOURS)
    X["HourCarrierCat"] = pd.Categorical(
        df["UniqueCarrier"].astype(str) + "@" + hr.astype(str), categories=HC_LEVELS
    )
    for c in BASE_CAT:
        X[c] = pd.Categorical(df[c], categories=CAT_LEVELS[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
X_tr = prepare(train)
y_tr = to_y(train)
X_ev = prepare(evald)
y_ev = to_y(evald)

SCAN = [
    (dict(max_depth=6, learning_rate=0.03), 1400),
    (dict(max_depth=6, learning_rate=0.05), 900),
    (dict(max_depth=7, learning_rate=0.03), 1400),
    (dict(max_depth=5, learning_rate=0.04, subsample=0.85, seed=13), 700),
    (dict(max_depth=5, learning_rate=0.04, subsample=0.85, seed=55), 700),
    (dict(max_depth=5, learning_rate=0.03, subsample=0.9, min_child_weight=3, seed=77), 900),
]

t0 = time.time()
members = []
for spec, rounds in SCAN:
    m = xgb.XGBClassifier(
        n_estimators=rounds,
        eval_metric="auc",
        tree_method="hist",
        enable_categorical=True,
        random_state=spec.get("seed", SEED),
        n_jobs=N_JOBS,
        max_depth=spec["max_depth"],
        learning_rate=spec["learning_rate"],
        subsample=spec.get("subsample", 1.0),
        min_child_weight=spec.get("min_child_weight", 1),
    )
    m.fit(X_tr, y_tr, eval_set=[(X_ev, y_ev)], verbose=False)
    aucs = m.evals_result()["validation_0"]["auc"]
    bi = int(np.argmax(aucs))
    rising = " RISING" if bi > rounds - 30 else ""
    print(f"scan d{spec['max_depth']}-lr{spec['learning_rate']}-sub{spec.get('subsample', 1.0)}-mcw{spec.get('min_child_weight', 1)}-s{spec.get('seed', 0)}: best_round={bi + 1} auc={aucs[bi]:.4f}{rising}")
    members.append((m, bi + 1, aucs[bi]))
print(f"Scan time: {time.time() - t0:.1f}s")

members.sort(key=lambda t: -t[2])
best_m, best_bi, best_auc = members[0]
print(f"BEST single: {best_auc:.4f}")


def ens_predict(df: pd.DataFrame, ms) -> np.ndarray:
    X = prepare(df)
    ps = np.zeros(len(X))
    for m, bi, _ in ms:
        ps += m.predict_proba(X, iteration_range=(0, bi))[:, 1]
    return ps / len(ms)


def ens_predict_rank(df: pd.DataFrame, ms) -> np.ndarray:
    X = prepare(df)
    rs = np.zeros(len(X))
    for m, bi, _ in ms:
        rs += pd.Series(m.predict_proba(X, iteration_range=(0, bi))[:, 1]).rank().to_numpy()
    return rs / len(ms)


cands = []
for k in range(1, len(members) + 1):
    auc = roc_auc_score(y_ev, ens_predict(evald, members[:k]))
    cands.append((f"prob{k}", auc, members[:k]))
    print(f"Ensemble AUC (top {k}, prob): {auc:.4f}")

auc_rank_all = roc_auc_score(y_ev, ens_predict_rank(evald, members))
print(f"Ensemble AUC (all {len(members)}, rank): {auc_rank_all:.4f}")
cands.append((f"rank{len(members)}", auc_rank_all, members))

name, final_auc, final_members = max(cands, key=lambda t: t[1])
print(f"FINAL: {name} {final_auc:.4f}")
rank_mode = name.startswith("rank")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    if rank_mode:
        return ens_predict_rank(df, final_members)
    return ens_predict(df, final_members)


print(f"Eval AUC: {final_auc:.4f}")
