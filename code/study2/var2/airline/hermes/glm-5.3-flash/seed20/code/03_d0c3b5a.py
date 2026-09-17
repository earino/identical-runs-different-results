"""XGBoost binary classifier on the airline delay task.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).
  3. All feature engineering lives in prepare(); encoders/statistics are fit on train only.
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

# columns known in advance from the task schema (fixed, not data-derived)
CATS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
y_train = (train[TARGET] == POSITIVE).astype(int)
GLOB = float(y_train.mean())
TE_SMOOTH = 20.0


def _fit_te(keys: pd.Series) -> pd.Series:
    """Smoothed mean-delay per key, fit on TRAIN rows only. Returns a mapping."""
    g = pd.DataFrame({"k": keys, "y": y_train}).groupby("k")["y"].agg(["sum", "count"])
    return (g["sum"] + TE_SMOOTH * GLOB) / (g["count"] + TE_SMOOTH)


def _apply_te(mapping: pd.Series, values: pd.Series) -> pd.Series:
    return values.map(mapping).fillna(GLOB).astype(float)


def _key(df: pd.DataFrame, parts: list[str]) -> pd.Series:
    """Composite key from a df's own columns."""
    s = df[parts[0]].astype(str)
    for p in parts[1:]:
        s = s + "|" + df[p].astype(str)
    return s


# fit all encodings on train once at import (stats from train only)
_TE = {
    "Month": _fit_te(train["Month"]),
    "DayofMonth": _fit_te(train["DayofMonth"]),
    "DayOfWeek": _fit_te(train["DayOfWeek"]),
    "UniqueCarrier": _fit_te(train["UniqueCarrier"]),
    "Origin": _fit_te(train["Origin"]),
    "Dest": _fit_te(train["Dest"]),
    "route": _fit_te(train["Origin"] + "-" + train["Dest"]),
    "origin|dow": _fit_te(train["Origin"] + "|" + train["DayOfWeek"]),
    "dest|dow": _fit_te(train["Dest"] + "|" + train["DayOfWeek"]),
    "carrier|hour": _fit_te(train["UniqueCarrier"] + "|" + (train["DepTime"] // 100).astype(str)),
    "origin|hour": _fit_te(train["Origin"] + "|" + (train["DepTime"] // 100).astype(str)),
}
_FREQ = {c: train[c].value_counts() for c in CATS}


def _dep_parts(d: pd.DataFrame) -> pd.DataFrame:
    """Decompose scheduled departure time; raw hhmm int splits poorly."""
    t = d["DepTime"].astype("int64")
    hr = t // 100
    return pd.DataFrame(
        {
            "dep_hour": hr,
            "dep_min": t % 100,
            "dep_5min": (t // 5) % 288,
            "dep_sin": np.sin(2 * np.pi * hr / 24.0),
            "dep_cos": np.cos(2 * np.pi * hr / 24.0),
        },
        index=d.index,
    )


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["month_te"] = _apply_te(_TE["Month"], df["Month"])
    X["dom_te"] = _apply_te(_TE["DayofMonth"], df["DayofMonth"])
    X["dow_te"] = _apply_te(_TE["DayOfWeek"], df["DayOfWeek"])
    X["carrier_te"] = _apply_te(_TE["UniqueCarrier"], df["UniqueCarrier"])
    X["origin_te"] = _apply_te(_TE["Origin"], df["Origin"])
    X["dest_te"] = _apply_te(_TE["Dest"], df["Dest"])
    X["route_te"] = _apply_te(_TE["route"], _key(df, ["Origin", "Dest"]))
    X["origin_dow_te"] = _apply_te(_TE["origin|dow"], _key(df, ["Origin", "DayOfWeek"]))
    X["dest_dow_te"] = _apply_te(_TE["dest|dow"], _key(df, ["Dest", "DayOfWeek"]))
    X["carrier_hour_te"] = _apply_te(
        _TE["carrier|hour"], _key(df, ["UniqueCarrier"]) + "|" + (df["DepTime"] // 100).astype(str)
    )
    X["origin_hour_te"] = _apply_te(
        _TE["origin|hour"], _key(df, ["Origin"]) + "|" + (df["DepTime"] // 100).astype(str)
    )

    for c in CATS:
        X[f"{c}_freq"] = df[c].map(_FREQ[c]).fillna(0.0).astype(float)

    X = pd.concat([X, _dep_parts(df)], axis=1)
    X["distance"] = df["Distance"].astype(float)
    X["dist_log"] = np.log1p(df["Distance"].astype(float))

    dow = df["DayOfWeek"].str[2:].astype(int)
    X["dow_int"] = dow.astype(float)
    X["month_int"] = df["Month"].str[2:].astype(float)
    X["dom_int"] = df["DayofMonth"].str[2:].astype(float)
    X["weekend"] = (dow >= 6).astype(float)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=800,
    max_depth=8,
    learning_rate=0.04,
    subsample=0.9,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_lambda=1.0,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
    eval_metric="auc",
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
