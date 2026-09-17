"""Baseline XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

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
TE_COLS = ["UniqueCarrier", "Origin", "Dest"]
SMOOTH = 20
y_int = (train[TARGET] == POSITIVE).astype(int)
PRIOR = float(y_int.mean())

# full-data target-encoding maps (train only) used at predict time
te_maps = {}
for c in TE_COLS:
    st = pd.DataFrame({"k": train[c].astype(str), "y": y_int}).groupby("k")["y"].agg(["sum", "count"])
    te_maps[c] = ((st["sum"] + SMOOTH * PRIOR) / (st["count"] + SMOOTH))


def _cnum(s):
    return pd.to_numeric(s.astype(str).str.replace("c-", "", regex=False), errors="coerce")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    X["month_num"] = _cnum(df["Month"])
    X["dom_num"] = _cnum(df["DayofMonth"])
    X["dow_num"] = _cnum(df["DayOfWeek"])
    dt = df["DepTime"].astype(float)
    hour = np.floor(dt / 100) % 24
    X["dep_time"] = dt.to_numpy()
    X["minute_of_day"] = (hour * 60 + (dt - np.floor(dt / 100) * 100)).to_numpy()
    X["distance"] = df["Distance"].astype(float).to_numpy()
    for c in TE_COLS:
        X[c + "_te"] = df[c].astype(str).map(te_maps[c]).fillna(PRIOR).to_numpy()
    return X


def prepare_train() -> pd.DataFrame:
    # training matrix with OUT-OF-FOLD target encoding (no leakage)
    X = prepare(train)
    rng = np.random.RandomState(SEED)
    fold = rng.randint(0, 5, len(train))
    keys = {c: train[c].astype(str).to_numpy() for c in TE_COLS}
    yv = y_int.to_numpy()
    for c in TE_COLS:
        col = np.full(len(train), PRIOR)
        for f in range(5):
            m = fold == f
            st = pd.DataFrame({"k": keys[c][~m], "y": yv[~m]}).groupby("k")["y"].agg(["sum", "count"])
            enc = (st["sum"] + SMOOTH * PRIOR) / (st["count"] + SMOOTH)
            mapped = pd.Series(keys[c][m]).map(enc).fillna(PRIOR).to_numpy()
            col[m] = mapped
        X[c + "_te"] = col
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=30,
    max_depth=6,
    learning_rate=0.1,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(prepare_train(), y_int.to_numpy())
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
