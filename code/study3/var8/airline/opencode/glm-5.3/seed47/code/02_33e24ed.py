"""XGBoost airline delay. Contract: prints `Eval AUC: 0.xxxx`; predict_proba(df) works on raw rows."""
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

# --- fitted on train only -----------------------------------------------------
_route_train = train["Origin"].astype(str) + "_" + train["Dest"].astype(str)
CAT_COLS = ["UniqueCarrier", "Origin", "Dest", "Route"]
cat_levels = {
    "UniqueCarrier": pd.Index(sorted(train["UniqueCarrier"].dropna().unique())),
    "Origin": pd.Index(sorted(train["Origin"].dropna().unique())),
    "Dest": pd.Index(sorted(train["Dest"].dropna().unique())),
    "Route": pd.Index(sorted(_route_train.unique())),
}


def _c_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype("string").str.extract(r"(\d+)")[0], errors="coerce")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["Month"] = _c_num(df["Month"])
    X["DayofMonth"] = _c_num(df["DayofMonth"])
    X["DayOfWeek"] = _c_num(df["DayOfWeek"])
    dep = pd.to_numeric(df["DepTime"], errors="coerce").fillna(0) % 2400
    X["DepTime"] = dep
    X["Hour"] = (dep // 100).astype(int)
    X["MinOfDay"] = (dep // 100) * 60 + (dep % 100)
    X["Distance"] = pd.to_numeric(df["Distance"], errors="coerce")
    X["LogDist"] = np.log1p(X["Distance"])
    X["UniqueCarrier"] = df["UniqueCarrier"]
    X["Origin"] = df["Origin"]
    X["Dest"] = df["Dest"]
    X["Route"] = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
rng = np.random.RandomState(SEED)
val_idx = rng.choice(len(train), size=10000, replace=False)
mask = np.ones(len(train), dtype=bool)
mask[val_idx] = False

FEATURE_SETS = {
    "noroute": [c for c in ["Month", "DayofMonth", "DayOfWeek", "DepTime", "Hour", "MinOfDay",
                            "Distance", "LogDist", "UniqueCarrier", "Origin", "Dest"]],
    "route": None,  # all
}


def make_model():
    return xgb.XGBClassifier(
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=7,
        min_child_weight=2,
        tree_method="hist",
        enable_categorical=True,
        eval_metric="auc",
        early_stopping_rounds=50,
        random_state=SEED,
        n_jobs=N_JOBS,
    )


Xtr = prepare(train)
results = {}
for name, cols in FEATURE_SETS.items():
    Xt = Xtr[mask] if cols is None else Xtr[mask][cols]
    Xv = Xtr[~mask] if cols is None else Xtr[~mask][cols]
    m = make_model()
    t0 = time.time()
    m.fit(Xt, to_y(train[mask]), eval_set=[(Xv, to_y(train[~mask]))], verbose=False)
    auc = roc_auc_score(to_y(train[~mask]), m.predict_proba(Xv)[:, 1])
    ev = roc_auc_score(to_y(evald), m.predict_proba(prepare(evald) if cols is None else prepare(evald)[cols])[:, 1])
    print(f"[{name}] valid_auc={auc:.4f} eval_auc={ev:.4f} best_iter={m.best_iteration} t={time.time()-t0:.1f}s")
    results[name] = (m, ev)

model = results["route"][0] if results["route"][1] >= results["noroute"][1] else results["noroute"][0]
USE_COLS = None if results["route"][1] >= results["noroute"][1] else FEATURE_SETS["noroute"]


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    if USE_COLS is not None:
        X = X[USE_COLS]
    return model.predict_proba(X)[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
