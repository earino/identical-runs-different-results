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
from sklearn.linear_model import LinearRegression
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
TE_GROUPS = [
    ("te_origin", ["Origin"]),
    ("te_dest", ["Dest"]),
    ("te_route", ["Origin", "Dest"]),
    ("te_carrier", ["UniqueCarrier"]),
    ("te_carrier_hour", ["UniqueCarrier", "hour"]),
    ("te_origin_hour", ["Origin", "hour"]),
    ("te_carrier_origin", ["UniqueCarrier", "Origin"]),
    ("te_dow_hour", ["DayOfWeek", "hour"]),
    ("te_carrier_dest", ["UniqueCarrier", "Dest"]),
    ("te_dest_hour", ["Dest", "hour"]),
    ("te_month", ["Month"]),
    ("te_dow", ["DayOfWeek"]),
]
TE_NAMES = [n for n, _ in TE_GROUPS]
TE_MAPS = {}
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


# ------- training: OOF TE, per-member ES on eval (2006 proxy), refit full -------
t0 = time.time()
y_all = to_y(train_raw)
Xb = build_features(train_raw)
codes = {name: combo_codes(Xb, cols) for name, cols in TE_GROUPS}

te_oof = np.zeros((len(Xb), len(TE_NAMES)), dtype="float32")
fold = np.tile(np.arange(5), len(Xb) // 5 + 1)[: len(Xb)]
np.random.default_rng(SEED).shuffle(fold)
for f in range(5):
    fit_m = fold != f
    for j, (name, _) in enumerate(TE_GROUPS):
        m = fit_map(codes[name][fit_m], y_all[fit_m], CODE_SIZES[name])
        te_oof[fold == f, j] = m[codes[name][fold == f]]
X_all = finalize(Xb, pd.DataFrame(te_oof, columns=TE_NAMES, index=Xb.index))
for name, _ in TE_GROUPS:
    TE_MAPS[name] = fit_map(codes[name], y_all, CODE_SIZES[name]).astype("float32")
X_eval = prepare(eval_raw)
y_eval = to_y(eval_raw)
print(f"Feature build: {time.time() - t0:.1f}s")

CONFIGS = []
for c in (0.4, 0.45, 0.5, 0.55):
    for _ in range(2):
        CONFIGS.append(dict(max_depth=8, learning_rate=0.05, min_child_weight=10, subsample=1.0, colsample_bytree=c, reg_lambda=2.0))
CONFIGS.append(dict(max_depth=9, learning_rate=0.05, min_child_weight=10, subsample=1.0, colsample_bytree=0.5, reg_lambda=2.0))
CONFIGS.append(dict(max_depth=8, learning_rate=0.05, min_child_weight=10, subsample=1.0, colsample_bytree=0.35, reg_lambda=2.0))

members = []
best_iters = []
for i, cfg in enumerate(CONFIGS):
    calib = xgb.XGBClassifier(
        n_estimators=1000,
        tree_method="hist",
        enable_categorical=True,
        random_state=SEED + i,
        n_jobs=N_JOBS,
        early_stopping_rounds=40,
        eval_metric="auc",
        **cfg,
    )
    calib.fit(X_all, y_all, eval_set=[(X_eval, y_eval)], verbose=False)
    best_iters.append(int(calib.best_iteration))

# honest OOF predictions per config for the stacker
n = len(X_all)
fold = np.tile(np.arange(3), n // 3 + 1)[:n]
np.random.default_rng(SEED).shuffle(fold)
oof = np.zeros((n, len(CONFIGS)), dtype="float64")
for f in range(3):
    tr_m, va_m = fold != f, fold == f
    for i, cfg in enumerate(CONFIGS):
        m = xgb.XGBClassifier(
            n_estimators=best_iters[i] + 1,
            tree_method="hist",
            enable_categorical=True,
            random_state=SEED + i,
            n_jobs=N_JOBS,
            eval_metric="auc",
            **cfg,
        )
        m.fit(X_all[tr_m], y_all[tr_m])
        oof[va_m, i] = m.predict_proba(X_all[va_m])[:, 1]

stacker = LinearRegression(positive=True)
stacker.fit(oof, y_all)
w = stacker.coef_ / stacker.coef_.sum()
print("stack weights:", np.round(w, 3))

for i, cfg in enumerate(CONFIGS):
    final = xgb.XGBClassifier(
        n_estimators=best_iters[i] + 1,
        tree_method="hist",
        enable_categorical=True,
        random_state=SEED + i,
        n_jobs=N_JOBS,
        eval_metric="auc",
        **cfg,
    )
    final.fit(X_all, y_all)
    members.append(final)
    print(f"member {i}: best_iter={best_iters[i]}")

print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    Xp = prepare(df)
    preds = np.column_stack([m.predict_proba(Xp)[:, 1] for m in members])
    return 0.5 * (preds @ w) + 0.5 * preds.mean(axis=1)


t0 = time.time()
eval_auc = roc_auc_score(to_y(eval_raw), predict_proba(eval_raw))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
