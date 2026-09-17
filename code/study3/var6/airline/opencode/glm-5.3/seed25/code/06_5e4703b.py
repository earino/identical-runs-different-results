"""XGBoost binary classifier. THIS IS THE FILE THE AGENT EDITS."""
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
CCLS = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CCLS}
# upweight late-2005 months: closer to the 2006 evaluation period
month_w = 1.0 + 2.0 * (train["Month"].str.slice(2).astype(int).to_numpy() - 1) / 11.0


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["DepTime"] = df["DepTime"].astype(float)
    X["Distance"] = df["Distance"].astype(float)
    X["hour"] = (df["DepTime"] // 100).astype(float)
    for c in CCLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- models: structurally diverse XGBoost ensemble ----------------------------
N_EST = 600
CONFIGS = [
    dict(max_depth=16, max_bin=1024),
    dict(max_depth=16, max_bin=2048),
    dict(max_depth=16, max_bin=1024, colsample_bynode=0.7),
]

t0 = time.time()
X_all, y_all = prepare(train), to_y(train)
models = []
for i, cfg in enumerate(CONFIGS):
    m = xgb.XGBClassifier(
        n_estimators=N_EST,
        learning_rate=0.05,
        tree_method="hist",
        enable_categorical=True,
        random_state=SEED + i,
        n_jobs=N_JOBS,
        **cfg,
    )
    m.fit(X_all, y_all, sample_weight=month_w)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s  ({len(models)} models)")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
