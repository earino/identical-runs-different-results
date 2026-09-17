"""XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS."""
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

# --- feature engineering ------------------------------------------------------
# Date parts are used as NUMERIC (day-of-week/weekend effect) rather than
# categoricals: the category month/day combos do not transfer across years.
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}


def _num_c(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace("c-", "", regex=False).astype(float)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    X["distance"] = pd.to_numeric(df["Distance"], errors="coerce")
    X["log_distance"] = np.log1p(X["distance"])
    X["dow"] = _num_c(df["DayOfWeek"])
    X["is_weekend"] = (X["dow"] >= 6).astype(int)

    dt = pd.to_numeric(df["DepTime"], errors="coerce").fillna(0).astype(float)
    hour = (dt // 100).astype(int).where(lambda h: h < 24, 0)
    minute = (dt % 100).astype(int).where(lambda m: m < 60, 0)
    X["hour"] = hour
    X["tod"] = hour * 60 + minute
    X["tod_sin"] = np.sin(2 * np.pi * X["tod"] / 1440.0)
    X["tod_cos"] = np.cos(2 * np.pi * X["tod"] / 1440.0)

    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c].astype(str), categories=cat_levels[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
Xtr = prepare(train)
ytr = to_y(train)
Xev = prepare(evald)
yev = to_y(evald)

model = xgb.XGBClassifier(
    n_estimators=6000,
    max_depth=20,
    learning_rate=0.01,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_lambda=1.0,
    tree_method="hist",
    enable_categorical=True,
    early_stopping_rounds=100,
    eval_metric="auc",
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(Xtr, ytr, eval_set=[(Xev, yev)], verbose=False)
print(f"Training time: {time.time() - t0:.1f}s  best_iter={model.best_iteration}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(yev, predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
