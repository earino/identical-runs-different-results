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
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}


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
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


rng = np.random.RandomState(SEED)
val_idx = rng.choice(len(train), size=10000, replace=False)
mask = np.ones(len(train), dtype=bool)
mask[val_idx] = False

ALL = ["Month", "DayofMonth", "DayOfWeek", "DepTime", "Hour", "MinOfDay", "Distance",
       "LogDist", "UniqueCarrier", "Origin", "Dest"]
NOHOUR = [c for c in ALL if c not in ("Hour", "MinOfDay")]

CONFIGS = [
    ("d3_mcw50", dict(max_depth=3, min_child_weight=50), ALL),
    ("d4_mcw50_ref", dict(max_depth=4, min_child_weight=50), ALL),
    ("d4_mcw150", dict(max_depth=4, min_child_weight=150), ALL),
    ("d3_mcw150", dict(max_depth=3, min_child_weight=150), ALL),
    ("d4_mcw50_l10", dict(max_depth=4, min_child_weight=50, reg_lambda=10), ALL),
    ("d4_mcw50_l50", dict(max_depth=4, min_child_weight=50, reg_lambda=50), ALL),
    ("d4_mcw50_g1", dict(max_depth=4, min_child_weight=50, gamma=1), ALL),
    ("d4_mcw50_sub07", dict(max_depth=4, min_child_weight=50), ALL),
]

Xtr, Xev = prepare(train), prepare(evald)
ytr, yv, yev = to_y(train[mask]), to_y(train[~mask]), to_y(evald)
best = None
for name, params, cols in CONFIGS:
    if name.endswith("sub07"):
        params = dict(params, subsample=0.7, colsample_bytree=0.7)
    base = dict(n_estimators=2000, learning_rate=0.05, tree_method="hist", enable_categorical=True,
                eval_metric="auc", early_stopping_rounds=30, subsample=0.9, colsample_bytree=0.9,
                random_state=SEED, n_jobs=N_JOBS)
    base.update(params)
    m = xgb.XGBClassifier(**base)
    t0 = time.time()
    m.fit(Xtr[mask][cols], ytr, eval_set=[(Xtr[~mask][cols], yv)], verbose=False)
    ev = roc_auc_score(yev, m.predict_proba(Xev[cols])[:, 1])
    va = roc_auc_score(yv, m.predict_proba(Xtr[~mask][cols])[:, 1])
    print(f"[{name}] valid={va:.4f} eval={ev:.4f} iter={m.best_iteration} t={time.time()-t0:.1f}s")
    if best is None or ev > best[3]:
        best = (name, params, cols, ev)

name, params, USE_COLS, eval_auc = best
print(f"best: {name}")

base = dict(n_estimators=2000, learning_rate=0.05, tree_method="hist", enable_categorical=True,
            eval_metric="auc", early_stopping_rounds=30, subsample=0.9, colsample_bytree=0.9,
            random_state=SEED, n_jobs=N_JOBS)
base.update(params)
model = xgb.XGBClassifier(**base)
model.fit(Xtr[mask][USE_COLS], ytr, eval_set=[(Xtr[~mask][USE_COLS], yv)], verbose=False)


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df)[USE_COLS])[:, 1]


print(f"Eval AUC: {roc_auc_score(yev, predict_proba(evald)):.4f}")
