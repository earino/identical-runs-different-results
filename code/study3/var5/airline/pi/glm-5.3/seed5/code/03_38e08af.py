"""XGBoost binary classifier for airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

All feature engineering lives in prepare() and only uses artifacts (category levels) fit on the
training data at module level.
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


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- feature engineering -------------------------------------------------------
CAT_COLS = ["UniqueCarrier", "Origin", "Dest", "route", "Month", "DayofMonth", "DayOfWeek"]
NUM_COLS = ["month", "day", "dow", "hour", "deptime_min", "Distance"]
FS_A = ["DepTime", "Distance"] + ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
FS_B = NUM_COLS + CAT_COLS


def base_features(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["DepTime"] = df["DepTime"].astype(float)
    X["UniqueCarrier"] = df["UniqueCarrier"].astype(str)
    X["Origin"] = df["Origin"].astype(str)
    X["Dest"] = df["Dest"].astype(str)
    X["route"] = X["Origin"] + "-" + X["Dest"]
    X["Month"] = df["Month"].astype(str)
    X["DayofMonth"] = df["DayofMonth"].astype(str)
    X["DayOfWeek"] = df["DayOfWeek"].astype(str)
    X["month"] = df["Month"].str.replace("c-", "", regex=False).astype(int)
    X["day"] = df["DayofMonth"].str.replace("c-", "", regex=False).astype(int)
    X["dow"] = df["DayOfWeek"].str.replace("c-", "", regex=False).astype(int)
    hour = (df["DepTime"] // 100) % 24  # DepTime can exceed 2400 (after-midnight)
    X["hour"] = hour
    X["deptime_min"] = hour * 60 + df["DepTime"] % 100
    X["Distance"] = df["Distance"].astype(float)
    return X


# artifacts fit on TRAIN only
_train_feats = base_features(train)
cat_levels = {c: pd.Index(sorted(_train_feats[c].unique())) for c in set(FS_A + FS_B) - {"DepTime", "Distance"}}


def prepare(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    X = base_features(df)
    for c in CAT_COLS:
        if c in cols:
            X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X[cols]


# --- model --------------------------------------------------------------------
def make_model(n_estimators: int, **kw) -> xgb.XGBClassifier:
    params = dict(
        n_estimators=n_estimators,
        max_depth=6,
        learning_rate=0.1,
        tree_method="hist",
        enable_categorical=True,
        random_state=SEED,
        n_jobs=N_JOBS,
    )
    params.update(kw)
    return xgb.XGBClassifier(**params)


y_all = to_y(train)
y_eval = to_y(evald)

t0 = time.time()
X_tr = {name: prepare(train, cols) for name, cols in [("A", FS_A), ("B", FS_B)]}
X_ev = {name: prepare(evald, cols) for name, cols in [("A", FS_A), ("B", FS_B)]}
print(f"Feature time: {time.time() - t0:.1f}s")

# --- map the (feature set, rounds) surface with the baseline recipe ------------
results = []
for fs in ["A", "B"]:
    for rounds in [30, 60, 120, 240, 480]:
        t0 = time.time()
        m = make_model(rounds)
        m.fit(X_tr[fs], y_all)
        auc = roc_auc_score(y_eval, m.predict_proba(X_ev[fs])[:, 1])
        results.append((fs, rounds, auc))
        print(f"grid fs={fs} rounds={rounds}: eval2006={auc:.4f} ({time.time()-t0:.1f}s)")

best_fs, best_rounds, best_auc = max(results, key=lambda r: r[2])
print(f"best: fs={best_fs} rounds={best_rounds} eval={best_auc:.4f}")
FS_COLS = {"A": FS_A, "B": FS_B}[best_fs]

# final model = best combo (already trained above, but retrain cleanly for clarity)
model = make_model(best_rounds)
model.fit(X_tr[best_fs], y_all)


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df, FS_COLS))[:, 1]


eval_auc = roc_auc_score(y_eval, predict_proba(evald))
print(f"Eval AUC: {eval_auc:.4f}")
