"""XGBoost binary classifier: airline delay. Feature engineering lives in prepare() only."""
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

# --- feature definitions (fitted on train only) --------------------------------
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {
    "UniqueCarrier": pd.Index(sorted(train["UniqueCarrier"].dropna().unique())),
    "Origin": pd.Index(sorted(train["Origin"].dropna().unique())),
    "Dest": pd.Index(sorted(train["Dest"].dropna().unique())),
}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    # calendar features as ordered ints (seasonality / weekly rhythm are monotone-ish signals)
    X["month"] = df["Month"].astype(str).str.slice(2).astype("int32")
    X["dayofmonth"] = df["DayofMonth"].astype(str).str.slice(2).astype("int32")
    X["dayofweek"] = df["DayOfWeek"].astype(str).str.slice(2).astype("int32")
    X["month_sin"] = np.sin(2 * np.pi * X["month"] / 12)
    X["month_cos"] = np.cos(2 * np.pi * X["month"] / 12)
    X["dow_sin"] = np.sin(2 * np.pi * X["dayofweek"] / 7)
    X["dow_cos"] = np.cos(2 * np.pi * X["dayofweek"] / 7)
    # scheduled departure time; DepTime can exceed 2400 (next-day red-eye slots) -> mod 24h
    dt = df["DepTime"].astype("int32")
    minutes = ((dt // 100) % 24) * 60 + (dt % 100).clip(0, 59)
    X["dep_minutes"] = minutes
    X["dep_hour"] = minutes // 60
    X["dep_sin"] = np.sin(2 * np.pi * minutes / 1440)
    X["dep_cos"] = np.cos(2 * np.pi * minutes / 1440)
    # distance
    X["distance"] = df["Distance"].astype("float32")
    X["distance_log"] = np.log1p(df["Distance"].astype("float32"))
    # categoricals (unseen levels -> NaN, handled natively by XGBoost)
    X["UniqueCarrier"] = pd.Categorical(df["UniqueCarrier"], categories=cat_levels["UniqueCarrier"])
    X["Origin"] = pd.Categorical(df["Origin"], categories=cat_levels["Origin"])
    X["Dest"] = pd.Categorical(df["Dest"], categories=cat_levels["Dest"])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
def _make(seed: int) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=2000, learning_rate=0.015, max_depth=0, max_leaves=160,
        min_child_weight=20, grow_policy="lossguide",
        subsample=0.85, colsample_bytree=0.4, reg_lambda=1.0, reg_alpha=4.0, gamma=2.0, tree_method="hist", max_bin=512,
        enable_categorical=True, random_state=seed, n_jobs=N_JOBS,
    )


models = [_make(s) for s in (SEED, SEED + 1, SEED + 2)]

t0 = time.time()
Xtr = prepare(train)
ytr = to_y(train)
for m in models:
    m.fit(Xtr, ytr)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
