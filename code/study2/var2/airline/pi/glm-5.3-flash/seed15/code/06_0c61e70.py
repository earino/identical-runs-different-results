"""XGBoost binary classifier on airline delays. Exp 5: FE + teH + hour-cat + carrier-x-hour cat."""
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

CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]

# ---- statistics fitted on train only -----------------------------------------
cat_levels = {c: pd.Index(sorted(train[c].astype(str).unique())) for c in CAT_COLS}
y_train = (train[TARGET] == POSITIVE).astype(int).to_numpy()
GLOBAL_MEAN = float(y_train.mean())
_k_h = 10
_h_tr = (train["DepTime"] // 100).astype(int).astype(str)
_h_ev = (evald["DepTime"] // 100).astype(int).astype(str)
_st_h = pd.DataFrame({"k": _h_tr, "y": y_train}).groupby("k")["y"].agg(["sum", "count"])
TE_H = (_st_h["sum"] + _k_h * GLOBAL_MEAN) / (_st_h["count"] + _k_h)


def _categorical(values: pd.Series, key: str) -> pd.Series:
    return pd.Categorical(values.astype(str), categories=cat_levels[key])


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering lives here (predict_proba calls this on unseen rows).
    X = pd.DataFrame(index=df.index)
    for c in CAT_COLS:
        X[c] = _categorical(df[c], c)
    X["Distance"] = df["Distance"].astype(float)
    X["Distance_log"] = np.log1p(df["Distance"].astype(float))
    s = df["DepTime"]
    hh = (s // 100).astype(int)
    mm = (s % 100).astype(int)
    minutes = hh * 60 + mm
    mod = minutes % 1440
    ang = 2 * np.pi * mod / 1440.0
    X["dep_minutes"] = minutes
    X["dep_hour"] = hh
    X["dep_sin"] = np.sin(ang)
    X["dep_cos"] = np.cos(ang)
    X["dep_wrap"] = (s >= 2400).astype(int)
    X["dep_mm"] = mm
    X["is00"] = (mm == 0).astype(int)
    X["is30"] = (mm == 30).astype(int)
    X["is45"] = (mm == 45).astype(int)
    X["Month_i"] = df["Month"].str[2:].astype(int)
    X["Dom_i"] = df["DayofMonth"].str[2:].astype(int)
    X["Dow_i"] = df["DayOfWeek"].str[2:].astype(int)
    mang = 2 * np.pi * (X["Month_i"] - 1) / 12.0
    X["month_sin"] = np.sin(mang)
    X["month_cos"] = np.cos(mang)
    X["is_weekend"] = (X["Dow_i"] >= 6).astype(int)
    hkey = hh.astype(str)
    X["teH"] = hkey.map(TE_H).fillna(GLOBAL_MEAN).astype(float).to_numpy()
    X["dep_hour_cat"] = pd.Categorical(hkey, categories=TE_H.index)
    X["car_h"] = pd.Categorical(
        df["UniqueCarrier"].astype(str) + "_" + hkey,
        categories=pd.Index(sorted((train["UniqueCarrier"].astype(str) + "_" + _h_tr).unique())),
    )
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model: 5-fold bag (every model sees only train.csv rows) -----------------
PARAMS = dict(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)
N_FOLDS = 5
PARAM_LISTS = [dict(p, max_depth=d) for d in (5, 6, 7) for p in [dict(n_estimators=300, learning_rate=0.1)]]

models = []
t0 = time.time()
X_all = prepare(train)
idx = np.arange(len(X_all))
for f in range(N_FOLDS):
    va = idx[f::N_FOLDS]
    tr = np.setdiff1d(idx, va)
    for j, base_params in enumerate(PARAM_LISTS):
        m = xgb.XGBClassifier(**{**PARAMS, **base_params, "random_state": SEED + f * 10 + j})
        m.fit(X_all.iloc[tr], y_train[tr])
        models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
