"""XGBoost binary classifier for airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract:
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. Module-level `predict_proba(df)`: raw DataFrame (target col may be absent) -> 1-D P(positive) array.
     ALL feature engineering lives inside prepare(); encoders are fit on training data only.
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
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- encoders / stats: fit on TRAIN ONLY ---------------------------------------
CAT_RAW = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].astype(str).unique())) for c in CAT_RAW}
route_levels = pd.Index(sorted((train["Origin"].astype(str) + "_" + train["Dest"].astype(str)).unique()))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["month"] = df["Month"].str.slice(2).astype(int)
    X["day"] = df["DayofMonth"].str.slice(2).astype(int)
    X["dow"] = df["DayOfWeek"].str.slice(2).astype(int)
    t = df["DepTime"].astype(int)
    X["hour"] = t // 100
    X["minute"] = t % 100
    X["dep_min"] = X["hour"] * 60 + X["minute"]
    X["distance"] = df["Distance"].astype(float)
    X["log_dist"] = np.log1p(X["distance"])
    X["carrier"] = pd.Categorical(df["UniqueCarrier"].astype(str), categories=cat_levels["UniqueCarrier"])
    X["origin"] = pd.Categorical(df["Origin"].astype(str), categories=cat_levels["Origin"])
    X["dest"] = pd.Categorical(df["Dest"].astype(str), categories=cat_levels["Dest"])
    X["route"] = pd.Categorical(df["Origin"].astype(str) + "_" + df["Dest"].astype(str), categories=route_levels)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model: time-ordered early stopping, then refit on full train --------------
PARAMS = dict(
    n_estimators=1000,
    max_depth=8,
    learning_rate=0.1,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
X_all = prepare(train)
y_all = to_y(train)
cut = int(len(train) * 0.8)  # last 20% of 2005 rows: emulates the 2005->2006 time shift
es = xgb.XGBClassifier(**PARAMS, early_stopping_rounds=50)
es.fit(X_all.iloc[:cut], y_all[:cut], eval_set=[(X_all.iloc[cut:], y_all[cut:])], verbose=False)
best_round = es.best_iteration + 1
print(f"ES fit: best_round={best_round}, val_auc={es.best_score:.4f}, {time.time() - t0:.1f}s")

model = xgb.XGBClassifier(**{**PARAMS, "n_estimators": best_round})
model.fit(X_all, y_all)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
