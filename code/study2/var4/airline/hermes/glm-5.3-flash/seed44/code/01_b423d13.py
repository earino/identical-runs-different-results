"""XGBoost binary classifier for the airline delay task.

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

FEATURE_COLS = [
    "sched_dep_time_min",
    "hour",
    "night",
    "month_num",
    "month_sin",
    "month_cos",
    "day_of_month",
    "day_of_week",
    "carrier",
    "origin",
    "dest",
    "route",
    "distance",
    "log_distance",
]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything
    # computed on train/eval outside this function would NOT be applied to the hidden holdout.
    # Per-feature encodings are fit on training data only (module-level, below) and referenced here.
    X = pd.DataFrame(index=df.index)
    dt = pd.to_numeric(df["DepTime"], errors="coerce")
    X["sched_dep_time_min"] = (dt // 100) * 60 + (dt % 100)
    hour = np.floor(X["sched_dep_time_min"] / 60)
    X["hour"] = np.clip(hour, 0, 23)  # hhmm > 2359 exists (e.g. 24xx/26xx): clamp into the day
    X["night"] = ((X["hour"] >= 21) | (X["hour"] <= 4)).astype(float)
    month_num = pd.to_numeric(df["Month"].str.removeprefix("c-"), errors="coerce").fillna(1.0)
    X["month_num"] = month_num
    ang = 2 * np.pi * (month_num - 1) / 12
    X["month_sin"] = np.sin(ang)
    X["month_cos"] = np.cos(ang)
    X["day_of_month"] = pd.to_numeric(
        df["DayofMonth"].str.removeprefix("c-"), errors="coerce"
    ).fillna(15.0)
    X["day_of_week"] = pd.to_numeric(
        df["DayOfWeek"].str.removeprefix("c-"), errors="coerce"
    ).fillna(4.0)
    X["carrier"] = df["UniqueCarrier"].astype("category")
    X["origin"] = df["Origin"].astype("category")
    X["dest"] = df["Dest"].astype("category")
    X["route"] = (df["Origin"].astype(str) + "_" + df["Dest"].astype(str)).astype("category")
    X["distance"] = pd.to_numeric(df["Distance"], errors="coerce")
    X["log_distance"] = np.log1p(X["distance"])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- category dictionaries (fit on training data only) -------------------------
_df_train_raw = pd.read_csv("data/train.csv")
CAT_FEATURES = {
    "carrier": _df_train_raw["UniqueCarrier"].astype(str).drop_duplicates().sort_values().tolist(),
    "origin": _df_train_raw["Origin"].astype(str).drop_duplicates().sort_values().tolist(),
    "dest": _df_train_raw["Dest"].astype(str).drop_duplicates().sort_values().tolist(),
    "route": (
        _df_train_raw["Origin"].astype(str) + "_" + _df_train_raw["Dest"].astype(str)
    ).drop_duplicates().sort_values().tolist(),
}
CAT_CODES = {c: {v: i for i, v in enumerate(vals)} for c, vals in CAT_FEATURES.items()}


def _apply_categories(X: pd.DataFrame) -> pd.DataFrame:
    for c in ("carrier", "origin", "dest", "route"):
        X[c] = (
            X[c].map(CAT_CODES[c]).astype("float64")
        )  # unseen levels -> NaN -> xgboost learns a default direction
    return X


model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=9,
    learning_rate=0.05,
    min_child_weight=8,
    subsample=0.9,
    colsample_bytree=0.7,
    reg_lambda=2.0,
    tree_method="hist",
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
train = _df_train_raw
evald = pd.read_csv("data/eval.csv")
model.fit(_apply_categories(prepare(train)), to_y(train))
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(_apply_categories(prepare(df)))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
