"""XGBoost binary classifier on airline delays. Experiment 2: FE v1 + early stopping."""
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
ROUTE = train["Origin"].astype(str) + "_" + train["Dest"].astype(str)

# fitted statistics (train only)
cat_levels = {c: pd.Index(sorted(train[c].astype(str).unique())) for c in CAT_COLS}
cat_levels["Route"] = pd.Index(sorted(ROUTE.unique()))
y_train = (train[TARGET] == POSITIVE).astype(int).to_numpy()
GLOBAL_MEAN = float(y_train.mean())


def _tod_parts(s: pd.Series) -> pd.DataFrame:
    hh = (s // 100).astype(int)
    mm = (s % 100).astype(int)
    minutes = hh * 60 + mm
    mod = minutes % 1440
    ang = 2 * np.pi * mod / 1440.0
    return pd.DataFrame({
        "dep_minutes": minutes,
        "dep_hour": hh,
        "dep_sin": np.sin(ang),
        "dep_cos": np.cos(ang),
        "dep_late_wrap": (s >= 2400).astype(int),
    })


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering lives here (predict_proba calls this on unseen rows).
    X = pd.DataFrame(index=df.index)
    # categoricals (unseen levels -> NaN handled by xgb categorical)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c].astype(str), categories=cat_levels[c])
    X["Route"] = pd.Categorical(
        df["Origin"].astype(str) + "_" + df["Dest"].astype(str), categories=cat_levels["Route"]
    )
    X["Distance"] = df["Distance"].astype(float)
    X["Distance_log"] = np.log1p(df["Distance"].astype(float))
    X = pd.concat([X, _tod_parts(df["DepTime"])], axis=1)
    # int date parts
    X["Month_i"] = df["Month"].str[2:].astype(int)
    X["Dom_i"] = df["DayofMonth"].str[2:].astype(int)
    X["Dow_i"] = df["DayOfWeek"].str[2:].astype(int)
    mang = 2 * np.pi * (X["Month_i"] - 1) / 12.0
    X["month_sin"] = np.sin(mang)
    X["month_cos"] = np.cos(mang)
    dang = 2 * np.pi * (X["Dom_i"] - 1) / 31.0
    X["dom_sin"] = np.sin(dang)
    X["dom_cos"] = np.cos(dang)
    X["is_weekend"] = (X["Dow_i"] >= 6).astype(int)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=3000,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    tree_method="hist",
    enable_categorical=True,
    early_stopping_rounds=100,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(
    prepare(train),
    y_train,
    eval_set=[(prepare(evald), (evald[TARGET] == POSITIVE).astype(int).to_numpy())],
    verbose=False,
)
print(f"Training time: {time.time() - t0:.1f}s  best_iter={model.best_iteration}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
