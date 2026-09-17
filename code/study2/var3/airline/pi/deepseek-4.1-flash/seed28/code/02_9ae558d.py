"""XGBoost binary classifier for airline delay prediction.

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

# --- global vocabularies / statistics: fit on TRAINING data only -----------------
CAT_SPECS = {
    "carrier": train["UniqueCarrier"],
    "origin": train["Origin"],
    "dest": train["Dest"],
}
CAT_LEVELS = {k: pd.Index(sorted(v.dropna().unique())) for k, v in CAT_SPECS.items()}


def _to_int(series: pd.Series) -> pd.Series:
    """c-<n> strings -> ints."""
    return series.astype(str).str.replace("c-", "", regex=False).astype(int)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """ALL feature engineering lives here so predict_proba() can reproduce it on unseen rows."""
    mon = _to_int(df["Month"])
    dom = _to_int(df["DayofMonth"])
    dow = _to_int(df["DayOfWeek"])

    dep = df["DepTime"].astype(int)
    tod = (dep // 100) * 60 + (dep % 100)  # minutes after midnight (may exceed 1440)
    tod = np.where(tod >= 1440, tod - 1440, tod)

    X = pd.DataFrame(index=df.index)
    X["month"] = mon.to_numpy()
    X["dom"] = dom.to_numpy()
    X["dow"] = dow.to_numpy()
    X["hour"] = (tod // 60).astype(np.float32)
    X["minute"] = (tod % 60).astype(np.float32)
    X["tod"] = tod.astype(np.float32)
    X["tod_sin"] = np.sin(2 * np.pi * tod / 1440.0).astype(np.float32)
    X["tod_cos"] = np.cos(2 * np.pi * tod / 1440.0).astype(np.float32)
    X["mon_sin"] = np.sin(2 * np.pi * mon / 12.0).astype(np.float32)
    X["mon_cos"] = np.cos(2 * np.pi * mon / 12.0).astype(np.float32)
    X["dow_sin"] = np.sin(2 * np.pi * dow / 7.0).astype(np.float32)
    X["dow_cos"] = np.cos(2 * np.pi * dow / 7.0).astype(np.float32)
    X["is_weekend"] = (dow >= 6).astype(np.int8)
    X["distance"] = df["Distance"].astype(np.float32)
    X["log_distance"] = np.log1p(df["Distance"].astype(np.float32))

    X["carrier"] = pd.Categorical(df["UniqueCarrier"], categories=CAT_LEVELS["carrier"])
    X["origin"] = pd.Categorical(df["Origin"], categories=CAT_LEVELS["origin"])
    X["dest"] = pd.Categorical(df["Dest"], categories=CAT_LEVELS["dest"])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
X_train = prepare(train)
y_train = to_y(train)

rng = np.random.RandomState(SEED)
perm = rng.permutation(len(X_train))
n_val = int(0.1 * len(X_train))
val_idx = perm[:n_val]
fit_idx = perm[n_val:]

model = xgb.XGBClassifier(
    n_estimators=3000,
    max_depth=8,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_lambda=1.0,
    tree_method="hist",
    enable_categorical=True,
    max_cat_to_onehot=1,
    random_state=SEED,
    n_jobs=N_JOBS,
    early_stopping_rounds=50,
)

t0 = time.time()
model.fit(
    X_train.iloc[fit_idx],
    y_train[fit_idx],
    eval_set=[(X_train.iloc[val_idx], y_train[val_idx])],
    verbose=False,
)
print(f"Training time: {time.time() - t0:.1f}s  best_iter={model.best_iteration}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
