"""XGBoost binary classifier for airline dep_delayed_15min. Only file the agent edits.

Contract: `python train.py` -> prints `Eval AUC: 0.xxxx`; module-level predict_proba(df) -> P(positive).
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

CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering lives here (called on unseen rows by predict_proba).
    X = pd.DataFrame(index=df.index)
    X["DepTime"] = df["DepTime"].astype("float64")
    X["Distance"] = df["Distance"].astype("float64")
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


rng = np.random.RandomState(0)
CONFIGS = []
for i in range(50):
    CONFIGS.append(
        dict(
            n_estimators=int(rng.choice([25, 30, 40])),
            max_depth=int(rng.choice([4, 5, 6, 6, 7, 8])),
            learning_rate=float(rng.choice([0.1, 0.1, 0.15])),
            subsample=float(rng.choice([0.7, 0.8, 0.9])),
            colsample_bytree=float(rng.choice([0.6, 0.7, 0.8, 0.9, 1.0])),
            min_child_weight=float(rng.choice([1.0, 5.0, 10.0])),
            reg_lambda=float(rng.choice([1.0, 1.0, 5.0])),
            gamma=float(rng.choice([0.0, 0.1])),
        )
    )


def make_model(seed: int, cfg: dict) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        **cfg,
        tree_method="hist",
        enable_categorical=True,
        random_state=seed,
        n_jobs=N_JOBS,
    )


t0 = time.time()
X_train, y_train = prepare(train), to_y(train)
models = [make_model(s, cfg).fit(X_train, y_train) for s, cfg in enumerate(CONFIGS, start=42)]
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
