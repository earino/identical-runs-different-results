"""XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

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
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
cat_cols = list(obj_cols)  # all string cols are low/moderate cardinality here
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}
# weekday schedule-type interactions: day-of-week x hour block
DOW_LEVELS = cat_levels["DayOfWeek"].astype(str)
DOW_SET = sorted(DOW_LEVELS.unique())


def hour_block(h: pd.Series) -> pd.Series:
    return pd.cut(h, bins=[-1, 5, 9, 15, 19, 24], labels=["night", "morning", "midday", "evening", "late"])


BLOCK_LEVELS = hour_block(pd.Series(range(24))).cat.categories
DOWxBLOCK = pd.Index([f"{d}_{b}" for d in DOW_SET for b in BLOCK_LEVELS])
DOWxHOUR = pd.Index([f"{d}_{h}" for d in DOW_SET for h in range(24)])
CARRxBLOCK = pd.Index([f"{c}_{b}" for c in cat_levels["UniqueCarrier"].astype(str) for b in BLOCK_LEVELS])


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    dep = X["DepTime"]
    hour = (dep // 100) % 24
    minute = dep % 100
    mins = hour * 60 + minute
    X["DepHour"] = hour
    X["DepMinute"] = minute
    X["DepMins"] = mins
    X["DepSin"] = np.sin(2 * np.pi * mins / 1440.0)
    X["DepCos"] = np.cos(2 * np.pi * mins / 1440.0)
    X["DistPerMin"] = X["Distance"] / (mins / 60.0 + 1.0)
    # scheduled time-of-day block and interactions
    blk = hour_block(hour)
    X["DepBlock"] = pd.Categorical(blk, categories=BLOCK_LEVELS)
    X["DOW_x_Block"] = pd.Categorical(
        X["DayOfWeek"].astype(str) + "_" + blk.astype(str),
        categories=DOWxBLOCK,
    )
    X["DOW_x_Hour"] = pd.Categorical(
        X["DayOfWeek"].astype(str) + "_" + hour.astype(str),
        categories=DOWxHOUR,
    )
    X["Carrier_x_Block"] = pd.Categorical(
        X["UniqueCarrier"].astype(str) + "_" + blk.astype(str),
        categories=CARRxBLOCK,
    )
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.1,
    max_bin=512,
    colsample_bytree=0.6,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(prepare(train), to_y(train))
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
