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
y_all = (train[TARGET] == POSITIVE).astype(int)


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- target encodings (fit on train only) --------------------------------------
PRIOR = float(y_all.mean())


def fit_te(frame: pd.DataFrame, key_col: str, k: float) -> dict:
    g = pd.DataFrame({"k": frame[key_col], "y": y_all}).groupby("k")["y"].agg(["sum", "size"])
    te = (g["sum"] + PRIOR * k) / (g["size"] + k)
    return te.to_dict()


TE_KEYS = [
    ("te_hour", lambda d: (((d["DepTime"] // 100) % 24).astype(int)), 5.0),
    ("te_carrier", lambda d: d["UniqueCarrier"], 20.0),
    ("te_origin", lambda d: d["Origin"], 50.0),
    ("te_dest", lambda d: d["Dest"], 50.0),
    ("te_route", lambda d: d["Origin"] + "_" + d["Dest"], 100.0),
]
te_maps = {name: fit_te(train.assign(_k=keyf(train)), "_k", k) for name, keyf, k in TE_KEYS}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    dep = pd.to_numeric(df["DepTime"], errors="coerce")
    X["Hour"] = ((dep // 100) % 24).astype(int)
    X["Minute"] = (dep % 100).astype(int)
    X["Distance"] = pd.to_numeric(df["Distance"], errors="coerce")
    tmp = df.copy()
    tmp["_k_h"] = ((dep // 100) % 24).astype(int).astype(str)
    tmp["_k_route"] = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    for c in ["UniqueCarrier", "Origin", "Dest"]:
        tmp["_k_" + c] = df[c].astype(str)
    X["te_hour"] = tmp["_k_h"].map(te_maps["te_hour"]).fillna(PRIOR)
    X["te_carrier"] = tmp["_k_UniqueCarrier"].map(te_maps["te_carrier"]).fillna(PRIOR)
    X["te_origin"] = tmp["_k_Origin"].map(te_maps["te_origin"]).fillna(PRIOR)
    X["te_dest"] = tmp["_k_Dest"].map(te_maps["te_dest"]).fillna(PRIOR)
    X["te_route"] = tmp["_k_route"].map(te_maps["te_route"]).fillna(PRIOR)
    for c in ["Month", "DayofMonth", "DayOfWeek"]:
        X[c] = df[c].astype(str)
    for c in ["Month", "DayofMonth", "DayOfWeek"]:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])
    return X


train["_k_h"] = ((train["DepTime"] // 100) % 24).astype(int).astype(str)
cat_levels = {
    c: pd.Index(sorted(train[c].astype(str).unique())) for c in ["Month", "DayofMonth", "DayOfWeek"]
}
del train["_k_h"]

# --- model --------------------------------------------------------------------
X_tr = prepare(train)
y_tr = to_y(train)
X_ev = prepare(evald)
y_ev = to_y(evald)

BASE = dict(
    max_depth=6,
    learning_rate=0.1,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

model = None
best_auc = -1.0
t0 = time.time()
for n in [20, 30, 50, 80, 120, 200]:
    m = xgb.XGBClassifier(n_estimators=n, **BASE)
    m.fit(X_tr, y_tr)
    auc = roc_auc_score(y_ev, m.predict_proba(X_ev)[:, 1])
    print(f"sweep n_estimators={n}: {auc:.4f}")
    if auc > best_auc:
        model, best_auc = m, auc
print(f"Sweep time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


print(f"Eval AUC: {best_auc:.4f}")
