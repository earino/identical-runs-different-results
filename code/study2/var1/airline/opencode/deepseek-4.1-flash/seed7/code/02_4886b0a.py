"""XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS."""
import json
import os
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

TASK = json.load(open("task.json"))
TARGET = TASK["target"]
POSITIVE = TASK["positive_label"]
ID_COLS = TASK.get("id_columns", [])
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42
PRIOR = 0.5
SMOOTH = 50.0

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}

TE_KEYS = ["UniqueCarrier", "Origin", "Dest", "route"]
train_route = train["Origin"].astype(str) + "_" + train["Dest"].astype(str)
y_all = (train[TARGET] == POSITIVE).astype(int).to_numpy()


def _key_series(df: pd.DataFrame, key: str) -> pd.Series:
    if key == "route":
        return df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    return df[key].astype(str)


def _te_fit(df: pd.DataFrame, y: np.ndarray) -> dict:
    maps = {}
    for key in TE_KEYS:
        g = pd.DataFrame({"k": _key_series(df, key).to_numpy(), "y": y}).groupby("k")["y"].agg(["sum", "count"])
        maps[key] = ((g["sum"] + PRIOR * SMOOTH) / (g["count"] + SMOOTH)).to_dict()
    return maps


te_maps = _te_fit(train, y_all)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])
    for key in TE_KEYS:
        X[f"te_{key}"] = _key_series(df, key).map(te_maps[key]).astype(float)
    return X


def _train_features() -> pd.DataFrame:
    X = prepare(train)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for tr_idx, va_idx in skf.split(X, y_all):
        m = _te_fit(train.iloc[tr_idx], y_all[tr_idx])
        for key in TE_KEYS:
            ks = _key_series(train.iloc[va_idx], key)
            X.loc[X.index[va_idx], f"te_{key}"] = ks.map(m[key]).astype(float).to_numpy()
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


model = xgb.XGBClassifier(
    n_estimators=60,
    max_depth=4,
    learning_rate=0.1,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(_train_features(), y_all)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
