"""XGBoost binary classifier — airline dep_delayed_15min. THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).
     All feature engineering lives inside prepare(); encoders are fit on training data only.
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

# --- feature engineering (encoders fit on training data only) ----------------------
FEATS_STR = ["DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
CAT_LEVELS = {c: pd.Index(sorted(train[c].dropna().unique())) for c in FEATS_STR}
HHCAR_CATS = pd.Index(sorted(((train["DepTime"] // 100).astype(str) + "|" + train["UniqueCarrier"].astype(str)).unique()))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """ALL feature engineering must be reachable from here (predict_proba path)."""
    X = pd.DataFrame(index=df.index)

    hh = df["DepTime"] // 100
    mm = df["DepTime"] % 100
    dep_norm = ((hh * 60 + mm) % 1440).astype(float)

    X["DepTime"] = df["DepTime"]
    X["dep_norm"] = dep_norm
    X["dep_sin"] = np.sin(2 * np.pi * dep_norm / 1440.0)
    X["dep_cos"] = np.cos(2 * np.pi * dep_norm / 1440.0)
    X["dep_late_night"] = ((hh >= 21) | (hh <= 5)).astype(int)
    X["dep_hh"] = hh
    X["Distance"] = df["Distance"]

    # interactions: scheduled-departure hour x carrier
    hh_s = hh.astype(str)
    keys = hh_s + "|" + df["UniqueCarrier"].astype(str)
    X["hhxcar"] = pd.Categorical(keys, categories=HHCAR_CATS)

    for c in FEATS_STR:
        X[c] = pd.Categorical(df[c], categories=CAT_LEVELS[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model -------------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=8,
    alpha=5,
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
