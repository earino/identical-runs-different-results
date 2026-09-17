"""XGBoost binary classifier for airline delay. Only file the agent edits.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).
     All feature engineering lives inside prepare(); encoders/stats are fit on training data only.
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

# --- feature layout -----------------------------------------------------------
CAT_COLS = ["UniqueCarrier", "Origin", "Dest", "Route"]
NUM_COLS = [
    "dep_hour", "time_min", "time_sin", "time_cos",
    "month_num", "month_sin", "month_cos",
    "dow_num", "dow_sin", "dow_cos",
    "dom_num", "dom_sin", "dom_cos",
    "Distance",
]
FEATURE_COLS = CAT_COLS + NUM_COLS

cat_levels = {c: pd.Index(sorted(train[c].dropna().astype(str).unique())) for c in CAT_COLS[:3]}
cat_levels["Route"] = pd.Index(sorted((train["Origin"].astype(str) + "_" + train["Dest"].astype(str)).unique()))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """ALL feature engineering lives here; fitted stats (cat_levels etc.) come from train only."""
    X = pd.DataFrame(index=df.index)
    # time of day from DepTime (hhmm; values >=2400 wrap past midnight)
    dt = pd.to_numeric(df["DepTime"], errors="coerce").fillna(0).astype(np.int32)
    hhmm = dt % 2400
    hour = hhmm // 100
    tmin = hour * 60 + hhmm % 100
    X["dep_hour"] = hour.astype(np.float32)
    X["time_min"] = tmin.astype(np.float32)
    ang = 2 * np.pi * tmin / 1440.0
    X["time_sin"] = np.sin(ang).astype(np.float32)
    X["time_cos"] = np.cos(ang).astype(np.float32)
    # calendar columns arrive as strings like "c-7"
    month = df["Month"].astype(str).str.slice(2).astype(np.int32)
    dom = df["DayofMonth"].astype(str).str.slice(2).astype(np.int32)
    dow = df["DayOfWeek"].astype(str).str.slice(2).astype(np.int32)
    X["month_num"] = month.astype(np.float32)
    a = 2 * np.pi * (month - 1) / 12.0
    X["month_sin"] = np.sin(a).astype(np.float32)
    X["month_cos"] = np.cos(a).astype(np.float32)
    X["dow_num"] = dow.astype(np.float32)
    a = 2 * np.pi * (dow - 1) / 7.0
    X["dow_sin"] = np.sin(a).astype(np.float32)
    X["dow_cos"] = np.cos(a).astype(np.float32)
    X["dom_num"] = dom.astype(np.float32)
    a = 2 * np.pi * (dom - 1) / 31.0
    X["dom_sin"] = np.sin(a).astype(np.float32)
    X["dom_cos"] = np.cos(a).astype(np.float32)
    X["Distance"] = pd.to_numeric(df["Distance"], errors="coerce").astype(np.float32)
    for c in CAT_COLS[:3]:
        X[c] = pd.Categorical(df[c].astype(str), categories=cat_levels[c])
    X["Route"] = pd.Categorical(
        df["Origin"].astype(str) + "_" + df["Dest"].astype(str), categories=cat_levels["Route"]
    )
    return X[FEATURE_COLS]


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
N_EST = 40
model = xgb.XGBClassifier(
    n_estimators=N_EST,
    max_depth=6,
    learning_rate=0.1,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(prepare(train), to_y(train))
print(f"Training time: {time.time() - t0:.1f}s")

imp = model.get_booster().get_score(importance_type="gain")
top = sorted(imp.items(), key=lambda kv: -kv[1])[:15]
print("top gain features:", top)


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
