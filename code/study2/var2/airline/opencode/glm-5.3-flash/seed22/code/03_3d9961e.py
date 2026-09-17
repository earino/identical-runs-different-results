"""XGBoost binary classifier for airline delays. THIS IS THE ONLY FILE THE AGENT EDITS.

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
cat_cols = [c for c in obj_cols if train[c].nunique() <= 5000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}
freq_maps = {c: train[c].value_counts(normalize=True) for c in ["UniqueCarrier", "Origin", "Dest"]}

cat_levels_all = dict(cat_levels)
cat_levels_all["Tod"] = pd.Index(["n1", "n2", "am", "day", "pm", "eve", "late"])


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    hour = (df["DepTime"] // 100) % 24
    minute = df["DepTime"] % 100
    t = hour * 60 + minute
    out = pd.DataFrame(
        {
            "DepTime": df["DepTime"],
            "Hour": hour,
            "SinT": np.sin(2 * np.pi * t / 1440.0),
            "CosT": np.cos(2 * np.pi * t / 1440.0),
            "Tod": pd.cut(t, bins=[-1, 359, 419, 599, 959, 1139, 1259, 1439],
                          labels=["n1", "n2", "am", "day", "pm", "eve", "late"]),
            "Distance": df["Distance"],
            "LogDist": np.log1p(df["Distance"]),
            "UniqueCarrier": df["UniqueCarrier"],
            "Origin": df["Origin"],
            "Dest": df["Dest"],
            "Month": df["Month"],
            "DayofMonth": df["DayofMonth"],
            "DayOfWeek": df["DayOfWeek"],
            "CarFreq": df["UniqueCarrier"].map(freq_maps["UniqueCarrier"]),
            "OrgFreq": df["Origin"].map(freq_maps["Origin"]),
            "DstFreq": df["Dest"].map(freq_maps["Dest"]),
        }
    )
    return out


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = add_features(df)
    for c in X.columns:
        if c in cat_levels_all:
            X[c] = pd.Categorical(X[c], categories=cat_levels_all[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
PARAMS = dict(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.04,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    enable_categorical=True,
    n_jobs=N_JOBS,
)
SEEDS = [1, 2, 3, 4, 5, 6, 7]

t0 = time.time()
X = prepare(train)
y = to_y(train)
models = []
for s in SEEDS:
    m = xgb.XGBClassifier(random_state=s, **PARAMS)
    m.fit(X, y)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    Xp = prepare(df)
    return np.mean([m.predict_proba(Xp)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
