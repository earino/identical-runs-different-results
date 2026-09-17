"""XGBoost binary classifier (airline delay). THIS IS THE ONLY FILE THE AGENT EDITS.

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

# --- feature definitions (fitted on TRAIN only, applied by prepare()) -----------
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}

# Frequency counts are fitted on train only. They act as a stable "size / busyness" proxy for carrier,
# airport and city-pair, which transfers across years far better than the identity categories alone.
FREQ_SRC = ["UniqueCarrier", "Origin", "Dest"]
_route_tr = train["Origin"].astype(str) + "_" + train["Dest"].astype(str)
freq_maps = {c: train[c].value_counts().astype(float) for c in FREQ_SRC}
freq_maps["route"] = _route_tr.value_counts().astype(float)

WEEKEND = {"c-6", "c-7"}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    # NOTE: the raw calendar columns (Month / DayofMonth / DayOfWeek) are deliberately NOT used as features:
    # their seasonal/day-of-month splits do not transfer from 2005 to later years, and dropping them is worth
    # ~+0.008 AUC on the (later-year) eval set.
    X = pd.DataFrame(index=df.index)
    X["Distance"] = pd.to_numeric(df["Distance"], errors="coerce").astype(float)
    X["log_dist"] = np.log1p(X["Distance"])
    # scheduled departure hhmm -> clock features (time of day is the strongest single signal)
    t = pd.to_numeric(df["DepTime"], errors="coerce").fillna(0.0)
    hour = (t // 100).clip(0, 23)
    minute = (t % 100).clip(0, 59)
    mod = hour * 60 + minute
    X["dep_hour"] = hour.astype(float)
    X["dep_minute"] = minute.astype(float)
    X["dep_sin"] = np.sin(2.0 * np.pi * mod / 1440.0)
    X["dep_cos"] = np.cos(2.0 * np.pi * mod / 1440.0)
    X["is_weekend"] = df["DayOfWeek"].isin(WEEKEND).astype(int)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    route = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    for c in FREQ_SRC:
        X[c + "_n"] = df[c].map(freq_maps[c]).fillna(0.0).astype(float)
    X["route_n"] = route.map(freq_maps["route"]).fillna(0.0).astype(float)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
# Bag of deep trees with strong column subsampling. Depth is the dominant knob on this dataset (the shallow
# baseline was badly underfit); the column subsample plus seed/depth averaging is what stops the deep trees
# from memorising 2005-specific noise.
PARAMS = dict(
    n_estimators=120,
    learning_rate=0.05,
    tree_method="hist",
    enable_categorical=True,
    n_jobs=N_JOBS,
)
MODEL_SPECS = [(14, 0.4), (20, 0.5), (24, 0.5), (32, 0.4)]

t0 = time.time()
models = []
for depth, colsample in MODEL_SPECS:
    for i in range(2):
        m = xgb.XGBClassifier(
            max_depth=depth,
            colsample_bytree=colsample,
            random_state=SEED + 100 * i,
            **PARAMS,
        )
        m.fit(prepare(train), to_y(train))
        models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
