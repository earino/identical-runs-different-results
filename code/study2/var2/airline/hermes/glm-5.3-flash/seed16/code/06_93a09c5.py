"""XGBoost binary classifier for airline delay (autoresearch). ONLY FILE EDITED.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Key rule: ALL feature engineering lives in prepare(df); any statistic used by it (maps, levels,
quantiles) is computed from training data only and stored in module state, so predict_proba()
reproduces it on unseen rows (e.g. the hidden holdout).
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

t00 = time.time()
train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- feature engineering (all inside prepare(); stats fitted on train only) ----
CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]


def _base_frame(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["DepTime"] = pd.to_numeric(df["DepTime"], errors="coerce")
    X["Distance"] = pd.to_numeric(df["Distance"], errors="coerce")
    for c in CAT_COLS:
        X[c] = df[c].astype("category")
    return X


def _fix_categories(X: pd.DataFrame) -> pd.DataFrame:
    for c in CAT_COLS:
        if str(X[c].dtype) != "category":
            X[c] = X[c].astype("category")
        X[c] = X[c].cat.set_categories(_cat_levels[c])
    return X


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = _base_frame(df)
    for c in CAT_COLS:
        X[c + "Freq"] = df[c].map(_freq_maps[c]).astype(float)  # unseen -> NaN, ok for trees
    X["RouteFreq"] = (df["Origin"].astype(str) + ">" + df["Dest"].astype(str)).map(_route_freq).astype(float)
    dt = X["DepTime"]
    hh = np.floor(dt / 100.0)
    mm = dt - hh * 100.0
    X["DepHour"] = hh + mm / 60.0
    X["DepHourSin"] = np.sin(2 * np.pi * X["DepHour"] / 24.0)
    X["DepHourCos"] = np.cos(2 * np.pi * X["DepHour"] / 24.0)
    X["DepMin"] = dt % 100
    X["DepTimeInt"] = dt
    X["LogDistance"] = np.log1p(X["Distance"])
    X["ShortHaul"] = (X["Distance"] < 400).astype(float)
    X["EarlyMorning"] = ((X["DepHour"] >= 5) & (X["DepHour"] < 7)).astype(float)
    X["LateNight"] = (X["DepHour"] >= 21).astype(float)
    X["AfternoonPeak"] = ((X["DepHour"] >= 15) & (X["DepHour"] < 20)).astype(float)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# fitted training statistics for predict_proba on unseen data
_cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}
_freq_maps = {c: train[c].value_counts().to_dict() for c in CAT_COLS}
_route_train = train["Origin"].astype(str) + ">" + train["Dest"].astype(str)
_route_freq = _route_train.value_counts().to_dict()

# --- model --------------------------------------------------------------------
MODEL_PARAMS = dict(
    n_estimators=1200,
    learning_rate=0.05,
    max_depth=9,
    min_child_weight=20,
    subsample=0.8,
    colsample_bytree=0.6,
    reg_lambda=5.0,
    reg_alpha=0.5,
    max_cat_to_onehot=10000,  # force native categorical splits (never one-hot)
    tree_method="hist",
    max_bin=512,
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
    eval_metric="auc",
    early_stopping_rounds=50,
)

# --- internal time-ordered validation (train is 2005; eval is 2006) -----------
n = len(train)
tr_part = train.iloc[: n - 20000]
va_part = train.iloc[n - 20000 :]
model = xgb.XGBClassifier(**MODEL_PARAMS)
model.fit(
    _fix_categories(prepare(tr_part)),
    to_y(tr_part),
    eval_set=[(_fix_categories(prepare(va_part)), to_y(va_part))],
    verbose=False,
)
print(f"best_iter={model.best_iteration} of {MODEL_PARAMS['n_estimators']}")
print(f"val_auc={model.best_score:.4f}")

# refit on all train with the best iteration count (early stopping off); seed ensemble
N_EST = int(model.best_iteration + 1)
P2 = {k: v for k, v in MODEL_PARAMS.items() if k != "early_stopping_rounds"}
P2["n_estimators"] = N_EST
X_full = _fix_categories(prepare(train))
y_full = to_y(train)
ENSEMBLE_SEEDS = [42, 43, 44, 45, 46]
models = []
for i, s_ in enumerate(ENSEMBLE_SEEDS):
    m = xgb.XGBClassifier(
        **{**P2, "random_state": s_, "max_depth": 8 + i, "colsample_bytree": [0.5, 0.6, 0.6, 0.7, 0.7][i]}
    )
    m.fit(X_full, y_full)
    models.append(m)


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = _fix_categories(prepare(df))
    ps = [m.predict_proba(X)[:, 1] for m in models]
    ranks = [pd.Series(p).rank(pct=True).to_numpy() for p in ps]  # rank-average: better for AUC
    return np.mean(ranks, axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
print(f"Total: {time.time() - t00:.1f}s")
