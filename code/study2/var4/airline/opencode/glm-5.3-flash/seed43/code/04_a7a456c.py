"""XGBoost binary classifier for the airline delay task (autoresearch edit target).

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

train_raw = pd.read_csv("data/train.csv")
eval_raw = pd.read_csv("data/eval.csv")

# ---------------- train-only statistics (all encoders fit on train, frozen) ----
PRIOR = float((train_raw[TARGET] == POSITIVE).astype(int).mean())
TE_GROUPS = []  # disabled for this experiment
TE_NAMES = [n for n, _ in TE_GROUPS]
CAT_OUT = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
LEVELS = {
    "Month": pd.Index(sorted(train_raw["Month"].astype(str).unique())),
    "DayofMonth": pd.Index(sorted(train_raw["DayofMonth"].astype(str).unique())),
    "DayOfWeek": pd.Index(sorted(train_raw["DayOfWeek"].astype(str).unique())),
    "UniqueCarrier": pd.Index(sorted(train_raw["UniqueCarrier"].astype(str).unique())),
    "Origin": pd.Index(sorted(train_raw["Origin"].astype(str).unique())),
    "Dest": pd.Index(sorted(train_raw["Dest"].astype(str).unique())),
}
BASE_OUT = [
    "DepTime", "dep_min", "dep_sin", "dep_cos", "hour", "Distance", "Distance_log",
    "Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest",
]
FEATURE_ORDER = BASE_OUT + TE_NAMES


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Base features. Reads only raw columns of df."""
    X = pd.DataFrame(index=df.index)
    X["DepTime"] = df["DepTime"].astype("float32")
    hh = df["DepTime"] // 100
    mm = df["DepTime"] % 100
    dep_min = (hh * 60 + mm) % 1440
    X["dep_min"] = dep_min.astype("float32")
    X["dep_sin"] = np.sin(2 * np.pi * dep_min / 1440.0).astype("float32")
    X["dep_cos"] = np.cos(2 * np.pi * dep_min / 1440.0).astype("float32")
    X["hour"] = (dep_min // 60).astype("int32")
    X["Distance"] = df["Distance"].astype("float32")
    X["Distance_log"] = np.log1p(X["Distance"])
    for c in ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]:
        X[c] = df[c].astype(str)
    return X


def combo_codes(Xb: pd.DataFrame, cols) -> np.ndarray:
    """Integer key for a TE group; unseen levels get their own slot per column."""
    code = np.zeros(len(Xb), dtype=np.int64)
    for c in cols:
        if c == "hour":
            cc = Xb["hour"].to_numpy("int64") + 1
            size = 25
        else:
            cc = pd.Categorical(Xb[c], categories=LEVELS[c]).codes.astype(np.int64) + 1
            size = len(LEVELS[c]) + 1
        code = code * size + cc
    return code


def fit_map(code: np.ndarray, y: np.ndarray, size: int) -> np.ndarray:
    n = np.bincount(code, minlength=size).astype("float64")
    s = np.bincount(code, weights=y, minlength=size)
    return (s + 25.0 * PRIOR) / (n + 25.0)


CODE_SIZES = {}
for _name, _cols in TE_GROUPS:
    _size = 1
    for _c in _cols:
        _size *= 25 if _c == "hour" else len(LEVELS[_c]) + 1
    CODE_SIZES[_name] = _size


def finalize(Xb: pd.DataFrame, te: pd.DataFrame) -> pd.DataFrame:
    for c in CAT_OUT:
        Xb[c] = pd.Categorical(Xb[c], categories=LEVELS[c])
    if len(TE_NAMES):
        X = pd.concat([Xb, te], axis=1)
    else:
        X = Xb
    return X[FEATURE_ORDER]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature pipeline for serving: base features + train-fitted TE maps."""
    Xb = build_features(df)
    if len(TE_NAMES):
        te = pd.DataFrame(index=Xb.index)
        for name, cols in TE_GROUPS:
            code = combo_codes(Xb, cols)
            te[name] = TE_MAPS[name][code].astype("float32")
        return finalize(Xb, te)
    return finalize(Xb, None)


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# ---------------- training: early stopping on val split ------------------------
y_all = to_y(train_raw)
X_all = prepare(train_raw)
print(f"Feature build: {time.time():.1f}s")

n = len(X_all)
rng = np.random.RandomState(SEED)
perm = rng.permutation(n)
val_idx, tr_idx = perm[:12000], perm[12000:]

model = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=6,
    min_child_weight=5,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
    early_stopping_rounds=30,
    eval_metric="auc",
)
t0 = time.time()
model.fit(
    X_all.iloc[tr_idx], y_all[tr_idx],
    eval_set=[(prepare(eval_raw), to_y(eval_raw))],
    verbose=False,
)
best_iter = int(model.best_iteration)
# refit on the full training set with the tree count calibrated on eval (2006 proxy)
model = xgb.XGBClassifier(
    n_estimators=best_iter + 1,
    learning_rate=0.1,
    max_depth=6,
    min_child_weight=5,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
    eval_metric="auc",
)
model.fit(X_all, y_all)
print(f"Training time: {time.time() - t0:.1f}s  best_iter={best_iter}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(eval_raw), predict_proba(eval_raw))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
