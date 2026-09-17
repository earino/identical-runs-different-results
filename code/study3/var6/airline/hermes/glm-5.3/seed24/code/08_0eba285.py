"""XGBoost binary classifier — airline dep-delay benchmark.

Contract (program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Feature engineering notes:
  - Target encodings are computed on train.csv only. Training rows get OUT-OF-FOLD
    values (5-fold, fit on the other folds) so the model never sees its own answer;
    any other DataFrame (eval, hidden holdout) gets the full-train statistic.
  - All row-level transforms live inside prepare(); module-level code only computes
    train-fit statistics.
"""
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

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- train-fit statistics ----------------------------------------------------
cat_levels = {
    "UniqueCarrier": pd.Index(sorted(train["UniqueCarrier"].dropna().unique())),
    "Origin": pd.Index(sorted(train["Origin"].dropna().unique())),
    "Dest": pd.Index(sorted(train["Dest"].dropna().unique())),
}

Y_TR = (train[TARGET] == POSITIVE).astype(int).to_numpy()
PRIOR = float(Y_TR.mean())
SMOOTH = 20.0


def _rate(sum_: pd.Series, count: pd.Series) -> pd.Series:
    return (sum_ + PRIOR * SMOOTH) / (count + SMOOTH)


def _full_map(keys: pd.Series) -> pd.Series:
    """key -> smoothed delay rate over the whole train."""
    g = pd.DataFrame({"k": keys.to_numpy(), "y": Y_TR}).groupby("k")["y"].agg(["sum", "count"])
    m = _rate(g["sum"], g["count"])
    m[-0.0] = PRIOR
    return m


def _oof(keys: pd.Series) -> pd.Series:
    """Out-of-fold smoothed delay rate per training row (indexed like train)."""
    oof = pd.Series(PRIOR, index=train.index, dtype="float64")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    k = keys.to_numpy()
    for tr_idx, val_idx in skf.split(k, Y_TR):
        g = pd.DataFrame({"k": k[tr_idx], "y": Y_TR[tr_idx]}).groupby("k")["y"].agg(["sum", "count"])
        m = _rate(g["sum"], g["count"])
        oof.iloc[val_idx] = pd.Series(k[val_idx]).map(m).fillna(PRIOR).to_numpy()
    return oof


_TR_HH = (train["DepTime"].astype("int32") // 100 % 24)

# key builders: one per target-encoded feature (all deterministic row->string)
KEYS = {
    "te_Origin": train["Origin"].astype(str),
    "te_Dest": train["Dest"].astype(str),
    "te_UniqueCarrier": train["UniqueCarrier"].astype(str),
    "te_hour": _TR_HH.astype(str),
    "te_hour_origin": train["Origin"].astype(str) + "@" + _TR_HH.astype(str),
    "te_hour_dest": train["Dest"].astype(str) + "@" + _TR_HH.astype(str),
    "te_route": train["Origin"].astype(str) + "_" + train["Dest"].astype(str),
    "te_carrier_hour": train["UniqueCarrier"].astype(str) + "@" + _TR_HH.astype(str),
}
FULL_MAPS = {name: _full_map(k) for name, k in KEYS.items()}
OOF_VALS = {name: _oof(k) for name, k in KEYS.items()}

feature_cols = ["Month", "DayofMonth", "DayOfWeek", "DepTime", "Distance",
                "UniqueCarrier", "Origin", "Dest", "sin_hour", "cos_hour",
                "min_of_day"] + list(KEYS.keys())


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    is_train = df is train  # training rows -> OOF values; anything else -> full-train maps
    X = pd.DataFrame(index=df.index)
    for c in ("Month", "DayofMonth", "DayOfWeek"):
        X[c] = df[c].map(lambda v: int(str(v).lstrip("c-"))).astype("int16")
    X["DepTime"] = df["DepTime"].astype("int32")
    X["Distance"] = df["Distance"].astype("float32")
    hh = df["DepTime"].astype("int32") // 100
    mm = df["DepTime"].astype("int32") % 100
    mins = np.minimum(hh, 23) * 60 + mm          # minutes since midnight, 24h+ clipped
    ang = 2 * np.pi * mins / (24 * 60.0)
    X["sin_hour"] = np.sin(ang).astype("float32")
    X["cos_hour"] = np.cos(ang).astype("float32")
    X["min_of_day"] = mins.astype("int32")
    hh2 = (hh % 24).astype("int8")
    _hh_s = hh2.astype(str)
    key_vals = {
        "te_Origin": df["Origin"].astype(str),
        "te_Dest": df["Dest"].astype(str),
        "te_UniqueCarrier": df["UniqueCarrier"].astype(str),
        "te_hour": _hh_s,
        "te_hour_origin": df["Origin"].astype(str) + "@" + _hh_s,
        "te_hour_dest": df["Dest"].astype(str) + "@" + _hh_s,
        "te_route": df["Origin"].astype(str) + "_" + df["Dest"].astype(str),
        "te_carrier_hour": df["UniqueCarrier"].astype(str) + "@" + _hh_s,
    }
    for name, kv in key_vals.items():
        if is_train:
            X[name] = OOF_VALS[name].to_numpy()
        else:
            X[name] = kv.map(FULL_MAPS[name]).astype("float32")
    for c in ("UniqueCarrier", "Origin", "Dest"):
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X[feature_cols]


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model -------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=4000,
    max_depth=8,
    learning_rate=0.03,
    tree_method="hist",
    enable_categorical=True,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    early_stopping_rounds=30,
    random_state=7,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(prepare(train), to_y(train), eval_set=[(prepare(evald), to_y(evald))], verbose=False)
print(f"Training time: {time.time() - t0:.1f}s  best_iteration={model.best_iteration}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df), iteration_range=(0, model.best_iteration + 1))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
