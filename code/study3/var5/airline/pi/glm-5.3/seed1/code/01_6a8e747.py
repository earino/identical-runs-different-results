"""XGBoost binary classifier — airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

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

# --- features -----------------------------------------------------------------
CAT_COLS = ["UniqueCarrier", "Origin", "Dest", "Route"]
cat_levels = {}
for c in ["UniqueCarrier", "Origin", "Dest"]:
    cat_levels[c] = pd.Index(sorted(train[c].dropna().unique()))
route_train = train["Origin"].astype(str) + "_" + train["Dest"].astype(str)
cat_levels["Route"] = pd.Index(sorted(route_train.unique()))


def _cnum(s: pd.Series) -> pd.Series:
    """'c-<n>' string column -> float n."""
    return pd.to_numeric(s.astype(str).str.replace("c-", "", regex=False), errors="coerce")


def _cyc(x: pd.Series, period: float) -> pd.DataFrame:
    return pd.DataFrame({f"{x.name}_sin": np.sin(2 * np.pi * x / period),
                        f"{x.name}_cos": np.cos(2 * np.pi * x / period)})


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    month = _cnum(df["Month"])
    day = _cnum(df["DayofMonth"])
    dow = _cnum(df["DayOfWeek"])
    X["Month"] = month
    X["DayofMonth"] = day
    X["DayOfWeek"] = dow
    X["DayOfYear"] = (month - 1) * 31 + day
    dep = pd.to_numeric(df["DepTime"], errors="coerce")
    hour = (dep // 100).astype(float)
    minute = dep % 100
    X["Hour"] = hour
    X["Minute"] = minute
    X["DepMinutes"] = hour * 60 + minute
    X["IsOver2400"] = (dep >= 2400).astype(int)
    X = pd.concat([X, _cyc(hour, 24.0)], axis=1)
    X["DowSin"], X["DowCos"] = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)
    X["MonthSin"], X["MonthCos"] = np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12)
    X["Distance"] = pd.to_numeric(df["Distance"], errors="coerce")
    X["LogDistance"] = np.log1p(X["Distance"])
    X["Route"] = (df["Origin"].astype(str) + "_" + df["Dest"].astype(str)).values
    for c in ["UniqueCarrier", "Origin", "Dest"]:
        X[c] = df[c].values
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
rng = np.random.RandomState(SEED)
idx = rng.permutation(len(train))
n_va = int(0.2 * len(train))
va_idx, tr_idx = idx[:n_va], idx[n_va:]

X_all = prepare(train)
y_all = to_y(train)

model = xgb.XGBClassifier(
    n_estimators=1500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    enable_categorical=True,
    early_stopping_rounds=50,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(X_all.iloc[tr_idx], y_all[tr_idx], eval_set=[(X_all.iloc[va_idx], y_all[va_idx])], verbose=False)
print(f"Training time: {time.time() - t0:.1f}s, best iter: {model.best_iteration}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
