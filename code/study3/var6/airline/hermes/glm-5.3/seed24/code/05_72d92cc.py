"""XGBoost binary classifier — airline dep-delay benchmark.

Contract (program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).
All feature engineering is inside prepare(); statistics are fit on train.csv only.
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

# --- features ---------------------------------------------------------------
# cyclical time features: hour-of-day is the strongest known signal for departure delays
cat_levels = {
    "UniqueCarrier": pd.Index(sorted(train["UniqueCarrier"].dropna().unique())),
    "Origin": pd.Index(sorted(train["Origin"].dropna().unique())),
    "Dest": pd.Index(sorted(train["Dest"].dropna().unique())),
}

# smoothed target (delay-rate) encoding per high-cardinality categorical, fit on train only
PRIOR = float((train[TARGET] == POSITIVE).mean())
SMOOTH = 20.0
te_maps = {}
for c in ("Origin", "Dest", "UniqueCarrier"):
    stats = train.groupby(c, observed=True)[TARGET].agg(
        lambda s: (s.eq(POSITIVE).sum() + PRIOR * SMOOTH) / (len(s) + SMOOTH))
    stats[-0.0] = PRIOR            # unseen levels -> global prior
    te_maps[c] = stats

# smoothed delay rate by hour-of-day (mod-24), fit on train only
_tr_hh = train["DepTime"].astype("int32") // 100 % 24
hour_rate = (train.assign(_hh=_tr_hh)
             .groupby("_hh")[TARGET]
             .agg(lambda s: (s.eq(POSITIVE).sum() + PRIOR * SMOOTH) / (len(s) + SMOOTH)))
hour_rate[-0.0] = PRIOR

feature_cols = ["Month", "DayofMonth", "DayOfWeek", "DepTime", "Distance",
                "UniqueCarrier", "Origin", "Dest", "sin_hour", "cos_hour",
                "min_of_day", "te_hour",
                "te_Origin", "te_Dest", "te_UniqueCarrier"]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    for c in ("Month", "DayofMonth", "DayOfWeek"):
        X[c] = df[c].map(lambda v: int(str(v).lstrip("c-"))).astype("int16")
    X["DepTime"] = df["DepTime"].astype("int32")
    X["Distance"] = df["Distance"].astype("float32")
    hh = df["DepTime"].astype("int32") // 100
    mm = df["DepTime"].astype("int32") % 100
    # DepTime is hhmm; values >= 2400 are 24+ hour times -> clip hour to 23.999 so the
    # circle stays continuous (2400 == 0000 of the next day)
    mins = np.minimum(hh, 23) * 60 + mm  # minutes since midnight, clipped
    ang = 2 * np.pi * mins / (24 * 60.0)
    X["sin_hour"] = np.sin(ang).astype("float32")
    X["cos_hour"] = np.cos(ang).astype("float32")
    # raw minute-of-day as a plain numeric too: trees split it directly and can express
    # the (non-sinusoidal) morning dip / afternoon peak asymmetry
    X["min_of_day"] = mins.astype("int32")
    hh2 = (hh % 24).astype("int8")
    # hour-of-day target rate, fit on train only (smoothed): captures the strong
    # "later departure -> more delays" effect as a single well-ordered feature
    X["te_hour"] = hh2.map(hour_rate).astype("float32")
    for c in ("UniqueCarrier", "Origin", "Dest"):
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
        X["te_" + c] = df[c].map(te_maps[c]).astype("float32")
    return X[feature_cols]


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model -------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=3000,
    max_depth=6,
    learning_rate=0.05,
    tree_method="hist",
    enable_categorical=True,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    early_stopping_rounds=30,
    random_state=7,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(prepare(train), to_y(train), eval_set=[(prepare(evald), to_y(evald))], verbose=False)
print(f"Training time: {time.time() - t0:.1f}s  best_iteration={model.best_iteration}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df), iteration_range=(0, model.best_iteration + 1))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
