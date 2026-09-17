"""XGBoost binary classifier for airline delays. THIS IS THE ONLY FILE THE AGENT EDITS.

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
from sklearn.model_selection import KFold

TASK = json.load(open("task.json"))
TARGET = TASK["target"]
POSITIVE = TASK["positive_label"]
ID_COLS = TASK.get("id_columns", [])
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- features -----------------------------------------------------------------
CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}

# target encodings, fitted on training data only
y_all = (train[TARGET] == POSITIVE).astype(float)
GMEAN = float(y_all.mean())
TE_SMOOTH = {"UniqueCarrier": 20, "Origin": 20, "Dest": 20, "Route": 10}


def _te_stats(keys: pd.Series, y: pd.Series, m: float) -> pd.DataFrame:
    g = pd.DataFrame({"k": keys.values, "y": y.values}).groupby("k")["y"].agg(["sum", "count"])
    return (g["sum"] + m * GMEAN) / (g["count"] + m)


train_keys = pd.DataFrame(
    {
        "UniqueCarrier": train["UniqueCarrier"],
        "Origin": train["Origin"],
        "Dest": train["Dest"],
        "Route": train["Origin"] + "_" + train["Dest"],
    }
)
full_te = {c: _te_stats(train_keys[c], y_all, m) for c, m in TE_SMOOTH.items()}
# out-of-fold target encodings for the training rows (no leakage into training fit)
oof_te = {c: pd.Series(np.nan, index=train.index) for c in TE_SMOOTH}
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
for tr_idx, va_idx in kf.split(train_keys):
    for c, m in TE_SMOOTH.items():
        st = _te_stats(train_keys[c].iloc[tr_idx], y_all.iloc[tr_idx], m)
        oof_te[c].iloc[va_idx] = train_keys[c].iloc[va_idx].map(st).fillna(GMEAN).values
cnt_full = {c: train_keys[c].value_counts() for c in ["Origin", "Dest", "Route"]}


def prepare(df: pd.DataFrame, te: dict | None = None) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    te_maps = te if te is not None else full_te
    X = pd.DataFrame(index=df.index)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    # scheduled departure time: hhmm, values > 2400 mean shortly after midnight next day
    dt = df["DepTime"].astype(float)
    hour = np.floor(dt / 100.0)
    minute = dt - hour * 100.0
    hour = np.where(hour >= 24, hour - 24, hour)
    mins = hour * 60.0 + minute
    X["DepTime"] = dt
    X["dep_hour"] = hour
    X["dep_minute"] = minute
    X["dep_minofday"] = mins
    ang = 2 * np.pi * mins / 1440.0
    X["dep_sin"] = np.sin(ang)
    X["dep_cos"] = np.cos(ang)
    dist = df["Distance"].astype(float)
    X["Distance"] = dist
    X["Distance_log"] = np.log1p(dist)
    # target / count encodings (statistics fitted on training data only)
    keys = pd.DataFrame(
        {
            "UniqueCarrier": df["UniqueCarrier"].values,
            "Origin": df["Origin"].values,
            "Dest": df["Dest"].values,
            "Route": df["Origin"].values + "_" + df["Dest"].values,
        }
    )
    for c, m in TE_SMOOTH.items():
        X["te_" + c] = keys[c].map(te_maps[c]).fillna(GMEAN).values
    for c in ["Origin", "Dest", "Route"]:
        X["cnt_" + c] = np.log1p(keys[c].map(cnt_full[c]).fillna(0).values)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
X_train = prepare(train, te=oof_te)
model.fit(X_train, to_y(train))
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
