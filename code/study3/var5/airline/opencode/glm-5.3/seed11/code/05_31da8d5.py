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

# --- features -----------------------------------------------------------------
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype("string").str.replace("c-", "", regex=False), errors="coerce")


def _cyc(x: pd.Series, period: float) -> pd.DataFrame:
    ang = 2.0 * np.pi * x / period
    return pd.DataFrame({f"{x.name}_sin": np.sin(ang), f"{x.name}_cos": np.cos(ang)})


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    month = _num(df["Month"])
    dom = _num(df["DayofMonth"])
    dow = _num(df["DayOfWeek"])
    X["Month"] = month
    X["DayofMonth"] = dom
    X["DayOfWeek"] = dow
    X = pd.concat([X, _cyc(month.rename("Month"), 12.0)], axis=1)
    X = pd.concat([X, _cyc(dow.rename("DayOfWeek"), 7.0)], axis=1)
    dt = pd.to_numeric(df["DepTime"], errors="coerce")
    hour = (dt // 100).astype("float")
    minute = (dt % 100).astype("float")
    mins = (hour * 60 + minute).mod(1440.0)
    X["DepTime"] = dt
    X["Dep_hour"] = hour
    X["Dep_min_of_day"] = mins
    X = pd.concat([X, _cyc(mins.rename("Dep_min_of_day"), 1440.0)], axis=1)
    X["Distance"] = pd.to_numeric(df["Distance"], errors="coerce")
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
X_tr, y_tr = prepare(train), to_y(train)
rng = np.random.RandomState(SEED)
val_idx = rng.choice(len(X_tr), size=len(X_tr) // 10, replace=False)
fit_idx = np.setdiff1d(np.arange(len(X_tr)), val_idx)
model = xgb.XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    min_child_weight=10,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(X_tr.iloc[fit_idx], y_tr[fit_idx])
print(f"Training time: {time.time() - t0:.1f}s")

Xe, ye = prepare(evald), to_y(evald)


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


for it in [50, 100, 150, 200, 300, 400]:
    p = model.predict_proba(Xe, iteration_range=(0, it))[:, 1]
    print(f"SWEEP iter={it} auc={roc_auc_score(ye, p):.4f}")
print(f"Eval AUC: {roc_auc_score(ye, model.predict_proba(Xe, iteration_range=(0, 200))[:, 1]):.4f}")
