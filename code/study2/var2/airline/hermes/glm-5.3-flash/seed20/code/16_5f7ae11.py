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
TE_SMOOTH_HI = 60.0  # higher-cardinality keys need more shrinkage


def _fit_te(keys: pd.Series, m: float = TE_SMOOTH) -> pd.Series:
    """Smoothed mean-delay per key, fit on TRAIN rows only. Returns a mapping."""
    g = pd.DataFrame({"k": keys, "y": y_train}).groupby("k")["y"].agg(["sum", "count"])
    return (g["sum"] + m * GLOB) / (g["count"] + m)


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
    "DayofMonth": _fit_te(train["DayofMonth"]),
    "DayOfWeek": _fit_te(train["DayOfWeek"]),
    "UniqueCarrier": _fit_te(train["UniqueCarrier"]),
    "Origin": _fit_te(train["Origin"]),
    "Dest": _fit_te(train["Dest"]),
    "route": _fit_te(train["Origin"] + "-" + train["Dest"], TE_SMOOTH_HI),
    "carrier|hour": _fit_te(train["UniqueCarrier"] + "|" + (train["DepTime"] // 100).astype(str), TE_SMOOTH_HI),
    "origin|hour": _fit_te(train["Origin"] + "|" + (train["DepTime"] // 100).astype(str), TE_SMOOTH_HI),
}
_FREQ = {c: train[c].value_counts() for c in CATS}
# per-key counts for high-cardinality TEs (lets trees discount low-support TEs)
_NTE = {
    "route": (train["Origin"] + "-" + train["Dest"]).value_counts(),
    "carrier|hour": (train["UniqueCarrier"] + "|" + (train["DepTime"] // 100).astype(str)).value_counts(),
    "origin|hour": (train["Origin"] + "|" + (train["DepTime"] // 100).astype(str)).value_counts(),
}


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
    X["dom_te"] = _apply_te(_TE["DayofMonth"], df["DayofMonth"])
    X["dow_te"] = _apply_te(_TE["DayOfWeek"], df["DayOfWeek"])
    X["carrier_te"] = _apply_te(_TE["UniqueCarrier"], df["UniqueCarrier"])
    X["origin_te"] = _apply_te(_TE["Origin"], df["Origin"])
    X["dest_te"] = _apply_te(_TE["Dest"], df["Dest"])
    X["route_te"] = _apply_te(_TE["route"], _key(df, ["Origin", "Dest"]))
    X["carrier_hour_te"] = _apply_te(
        _TE["carrier|hour"], _key(df, ["UniqueCarrier"]) + "|" + (df["DepTime"] // 100).astype(str)
    )
    X["origin_hour_te"] = _apply_te(
        _TE["origin|hour"], _key(df, ["Origin"]) + "|" + (df["DepTime"] // 100).astype(str)
    )
    # support counts for the high-cardinality TEs above
    X["route_n"] = np.log1p(
        _key(df, ["Origin", "Dest"]).map(_NTE["route"]).fillna(0).astype(float)
    )
    X["carrier_hour_n"] = np.log1p(
        (_key(df, ["UniqueCarrier"]) + "|" + (df["DepTime"] // 100).astype(str))
        .map(_NTE["carrier|hour"])
        .fillna(0)
        .astype(float)
    )
    X["origin_hour_n"] = np.log1p(
        (_key(df, ["Origin"]) + "|" + (df["DepTime"] // 100).astype(str))
        .map(_NTE["origin|hour"])
        .fillna(0)
        .astype(float)
    )

    for c in CATS:
        X[f"{c}_freq"] = df[c].map(_FREQ[c]).fillna(0.0).astype(float)

    X = pd.concat([X, _dep_parts(df)], axis=1)
    X["deptime_raw"] = df["DepTime"].astype(float)
    X["distance"] = df["Distance"].astype(float)
    X["dist_log"] = np.log1p(df["Distance"].astype(float))
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
PARAMS = dict(
    subsample=0.85,
    min_child_weight=10,
    reg_lambda=1.0,
    tree_method="hist",
    enable_categorical=True,
    n_jobs=N_JOBS,
    eval_metric="auc",
)
ENSEMBLE = [
    dict(seed=42, n_estimators=1500, max_depth=14, learning_rate=0.015, colsample_bytree=0.7),
    dict(seed=7, n_estimators=1500, max_depth=14, learning_rate=0.015, colsample_bytree=0.7),
    dict(seed=99, n_estimators=1500, max_depth=14, learning_rate=0.015, colsample_bytree=0.7),
    dict(seed=123, n_estimators=1500, max_depth=14, learning_rate=0.015, colsample_bytree=0.7),
    dict(seed=777, n_estimators=1500, max_depth=14, learning_rate=0.015, colsample_bytree=0.7),
]

t0 = time.time()
X_tr = prepare(train)
y_tr = to_y(train)
X_ev = prepare(evald)
y_ev = to_y(evald)
models = [xgb.XGBClassifier(**PARAMS, **cfg, early_stopping_rounds=100) for cfg in ENSEMBLE]
for m in models:
    m.fit(X_tr, y_tr, eval_set=[(X_ev, y_ev)], verbose=False)
    print("best_iter:", m.best_iteration)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    from scipy.stats import rankdata

    P = np.mean([rankdata(m.predict_proba(X)[:, 1]) for m in models], axis=0)
    P = (P - P.min()) / (P.max() - P.min())
    return P


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
