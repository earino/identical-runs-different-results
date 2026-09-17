"""XGBoost binary classifier — airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Recipe: smoothed target encoding of categorical keys (carrier, origin, dest, route,
route x hour, origin x hour, dest x hour) computed out-of-fold for the training matrix and
from full-train maps at predict time; numeric date/time features; XGBoost d8/mcw10.
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
SMOOTH = 50
y_int = (train[TARGET] == POSITIVE).astype(int)
PRIOR = float(y_int.mean())


def _hour(df):
    return (np.floor(df["DepTime"].astype(float) / 100) % 24).astype(int)


def _route_key(df):
    return df["Origin"].astype(str) + "_" + df["Dest"].astype(str)


def _keys(df):
    """All TE keys, built identically for train / eval / holdout."""
    h = _hour(df).astype(str)
    r = _route_key(df)
    return {
        "carrier": df["UniqueCarrier"].astype(str),
        "origin": df["Origin"].astype(str),
        "dest": df["Dest"].astype(str),
        "route": r,
        "route_hour": r + "_" + h,
        "origin_hour": df["Origin"].astype(str) + "_" + h,
        "dest_hour": df["Dest"].astype(str) + "_" + h,
    }


TRAIN_KEYS = _keys(train)


# full-data target-encoding maps (train only) used at predict time
def _te_map(key_series):
    st = pd.DataFrame({"k": key_series, "y": y_int}).groupby("k")["y"].agg(["sum", "count"])
    return (st["sum"] + SMOOTH * PRIOR) / (st["count"] + SMOOTH)


te_maps = {name: _te_map(s) for name, s in TRAIN_KEYS.items()}


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
    X["hour_sin"] = np.sin(2 * np.pi * X["minute_of_day"] / 1440)
    X["hour_cos"] = np.cos(2 * np.pi * X["minute_of_day"] / 1440)
    X["distance"] = df["Distance"].astype(float).to_numpy()
    keys = _keys(df)
    for name, series in keys.items():
        X[name + "_te"] = series.map(te_maps[name]).fillna(PRIOR).to_numpy()
    return X


def prepare_train() -> pd.DataFrame:
    # training matrix with OUT-OF-FOLD target encoding (no leakage)
    X = prepare(train)
    rng = np.random.RandomState(SEED)
    fold = rng.randint(0, 5, len(train))
    yv = y_int.to_numpy()
    for name, series in TRAIN_KEYS.items():
        col = np.full(len(train), PRIOR)
        kv = series.to_numpy()
        for f in range(5):
            m = fold == f
            st = pd.DataFrame({"k": kv[~m], "y": yv[~m]}).groupby("k")["y"].agg(["sum", "count"])
            enc = (st["sum"] + SMOOTH * PRIOR) / (st["count"] + SMOOTH)
            col[m] = pd.Series(kv[m]).map(enc).fillna(PRIOR).to_numpy()
        X[name + "_te"] = col
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=240,
    max_depth=8,
    min_child_weight=10,
    learning_rate=0.07,
    tree_method="hist",
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
