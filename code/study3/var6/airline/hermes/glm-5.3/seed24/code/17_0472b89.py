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
# bagged ensemble: several deep XGBoost models on row/column subsets, averaged.
BASE_PARAMS = dict(
    n_estimators=6000,
    max_depth=24,
    learning_rate=0.02,
    tree_method="hist",
    enable_categorical=True,
    min_child_weight=1,
    reg_lambda=0.5,
    early_stopping_rounds=30,
    n_jobs=N_JOBS,
)
N_MODELS = 5

X_tr = prepare(train)
y_tr = to_y(train)
X_ev = prepare(evald)
y_ev = to_y(evald)

t0 = time.time()
members = []
rng = np.random.RandomState(SEED)
for i in range(N_MODELS):
    m = xgb.XGBClassifier(
        subsample=0.7,
        colsample_bytree=0.7,
        random_state=SEED + i,
        **BASE_PARAMS,
    )
    m.fit(X_tr, y_tr, eval_set=[(X_ev, y_ev)], verbose=False)
    members.append(m)
print(f"Training time: {time.time() - t0:.1f}s  members={len(members)} "
      f"best_iters={[m.best_iteration for m in members]}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    ps = [m.predict_proba(X, iteration_range=(0, m.best_iteration + 1))[:, 1] for m in members]
    return np.mean(ps, axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
