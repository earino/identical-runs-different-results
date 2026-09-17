"""XGBoost binary classifier on the airline delay task.

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

CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}

# --- count encodings (fit on train only) ---------------------------------------
cnt_stats = {
    "Origin": train["Origin"].value_counts().to_dict(),
    "Dest": train["Dest"].value_counts().to_dict(),
    "route": (train["Origin"] + "_" + train["Dest"]).value_counts().to_dict(),
    "UniqueCarrier": train["UniqueCarrier"].value_counts().to_dict(),
}


def _to_int(s: pd.Series) -> np.ndarray:
    return s.str[2:].astype(int).to_numpy()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    month = _to_int(df["Month"])
    dom = _to_int(df["DayofMonth"])
    dow = _to_int(df["DayOfWeek"])
    dep = df["DepTime"].astype(int).to_numpy()
    hour = dep // 100
    minute = dep % 100
    X["month"] = month
    X["dom"] = dom
    X["dow"] = dow
    X["deptime"] = dep
    X["hour"] = hour
    X["minute"] = minute
    ang = 2 * np.pi * (hour + minute / 60.0) / 24.0
    X["hour_sin"] = np.sin(ang)
    X["hour_cos"] = np.cos(ang)
    X["dist"] = df["Distance"].to_numpy()
    X["dist_log"] = np.log1p(df["Distance"].to_numpy())
    cum = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])  # non-leap cum days
    doy = cum[np.clip(month - 1, 0, 11)] + dom
    X["is_winter_holiday"] = ((doy >= 349) | (doy <= 5)).astype(int)   # Dec 15 - Jan 5
    X["is_thanksgiving"] = ((doy >= 324) & (doy <= 334)).astype(int)   # Nov 20 - 30
    for c, stats in cnt_stats.items():
        if c == "route":
            keys = df["Origin"] + "_" + df["Dest"]
        else:
            keys = df[c]
        X["cnt_" + c] = np.log1p(keys.map(stats).fillna(0.0).to_numpy())
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=12000,
    max_depth=0,
    learning_rate=0.01,
    subsample=0.7,
    colsample_bytree=0.6,
    colsample_bynode=0.7,
    tree_method="hist",
    enable_categorical=True,
    early_stopping_rounds=300,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
X_tr = prepare(train)
y_tr = to_y(train)
rng = np.random.RandomState(SEED)
idx = rng.permutation(len(X_tr))
n_val = len(X_tr) // 5
val_idx, fit_idx = idx[:n_val], idx[n_val:]
model.fit(
    X_tr.iloc[fit_idx],
    y_tr[fit_idx],
    eval_set=[(X_tr.iloc[val_idx], y_tr[val_idx])],
    verbose=False,
)
print(f"Training time: {time.time() - t0:.1f}s, best_iter={model.best_iteration}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
