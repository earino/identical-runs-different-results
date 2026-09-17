"""XGBoost binary classifier for airline delay. Exp 10: 5-seed bagging ensemble."""
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
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}
ENG_COLS = ["DepHour", "DepTimeSin", "DepTimeCos", "DistanceLog"]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    t = pd.to_numeric(df["DepTime"], errors="coerce").astype("float64")
    t_mod = t % 2400  # 2400-2435 midnight-crossings -> 0-35
    hh = (t_mod // 100).clip(0, 23)
    mm = (t_mod % 100).clip(0, 59)
    frac = (hh * 60.0 + mm) / 1440.0
    X["DepHour"] = hh
    X["DepTimeSin"] = np.sin(2 * np.pi * frac)
    X["DepTimeCos"] = np.cos(2 * np.pi * frac)
    X["DistanceLog"] = np.log1p(pd.to_numeric(df["Distance"], errors="coerce").astype("float64"))
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
N_SEEDS = 5
XT = prepare(train)
YT = to_y(train)
XE = prepare(evald)
YE = to_y(evald)

t0 = time.time()
models = []
for s in range(N_SEEDS):
    m = xgb.XGBClassifier(
        n_estimators=8000,
        learning_rate=0.02,
        max_depth=6,
        min_child_weight=5.0,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        enable_categorical=True,
        early_stopping_rounds=200,
        eval_metric="auc",
        random_state=SEED + s,
        n_jobs=N_JOBS,
    )
    m.fit(XT, YT, eval_set=[(XE, YE)], verbose=False)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s best_iters={[m.best_iteration for m in models]}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    Xp = prepare(df)
    preds = [m.predict_proba(Xp)[:, 1] for m in models]
    return np.mean(preds, axis=0)


t0 = time.time()
eval_auc = roc_auc_score(YE, predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
