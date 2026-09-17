"""XGBoost binary classifier for airline delay prediction. THIS IS THE ONLY FILE THE AGENT EDITS.

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
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}

# --- target encodings (fit on TRAIN only; OOF for the training matrix) --------
TE_SMOOTH = 20.0

def _te_keys(df: pd.DataFrame) -> pd.DataFrame:
    k = pd.DataFrame(index=df.index)
    k["UniqueCarrier"] = df["UniqueCarrier"].astype(str)
    k["Origin"] = df["Origin"].astype(str)
    k["Dest"] = df["Dest"].astype(str)
    k["route"] = k["Origin"] + "_" + k["Dest"]
    k["hour"] = (df["DepTime"].astype(float) // 100).astype(int).astype(str)
    k["carrier_hour"] = k["UniqueCarrier"] + "_" + k["hour"]
    k["origin_hour"] = k["Origin"] + "_" + k["hour"]
    k["dest_hour"] = k["Dest"] + "_" + k["hour"]
    k["route_hour"] = k["route"] + "_" + k["hour"]
    k["carrier_origin"] = k["UniqueCarrier"] + "_" + k["Origin"]
    k["carrier_dest"] = k["UniqueCarrier"] + "_" + k["Dest"]
    return k

_te_train = _te_keys(train)
_y_all = (train[TARGET] == POSITIVE).astype(float).to_numpy()
_PRIOR = float(_y_all.mean())
TE_COLS = ["carrier_hour", "origin_hour", "dest_hour", "route_hour", "carrier_origin", "carrier_dest"]
PLAIN_TE = ["UniqueCarrier", "Origin", "Dest", "route", "hour"]

# full-train smoothed stats -> used for eval/holdout rows
team_full = {}
for c in TE_COLS:
    g = _te_train.assign(y=_y_all).groupby(c, sort=False)["y"].agg(["mean", "count"])
    team_full[c] = dict(zip(g.index, (g["mean"] * g["count"] + TE_SMOOTH * _PRIOR) / (g["count"] + TE_SMOOTH)))

# out-of-fold encodings for the training matrix (avoids leakage into the learner)
rng = np.random.default_rng(SEED)
folds = rng.permutation(len(_te_train)) % 5
X_oof = np.zeros((len(_te_train), len(TE_COLS)))
for f in range(5):
    tr_m, va_m = folds != f, folds == f
    for j, c in enumerate(TE_COLS):
        g = pd.DataFrame({c: _te_train[c].values[tr_m], "y": _y_all[tr_m]}).groupby(c)["y"].agg(["mean", "count"])
        enc = (g["mean"] * g["count"] + TE_SMOOTH * _PRIOR) / (g["count"] + TE_SMOOTH)
        X_oof[va_m, j] = pd.Series(_te_train[c].values[va_m]).map(enc).fillna(_PRIOR)
TE_OOF = pd.DataFrame(X_oof, columns=[f"te_{c}" for c in TE_COLS], index=train.index)

# --- frequency/congestion counts from train -------------------------------------
COUNT_COLS = ["UniqueCarrier", "Origin", "Dest", "route", "origin_hour", "dest_hour"]
cnt_full = {c: _te_train[c].value_counts().to_dict() for c in COUNT_COLS}

def _add_counts(X: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    keys = _te_keys(df)
    for c in COUNT_COLS:
        X[f"n_{c}"] = np.log1p(keys[c].map(cnt_full[c]).fillna(0.0).astype(float).values)
    return X

def _add_te(X: pd.DataFrame, df: pd.DataFrame, oof: pd.DataFrame | None) -> pd.DataFrame:
    keys = _te_keys(df)
    for c in TE_COLS:
        col = f"te_{c}"
        if oof is not None and len(oof) == len(X):
            X[col] = oof[col].values
        else:
            X[col] = keys[c].map(team_full[c]).astype(float)
    return X


def _parse_c(s: pd.Series) -> pd.Series:
    return s.astype(str).str.slice(2).astype(float)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    X["Month"] = _parse_c(df["Month"])
    X["DayofMonth"] = _parse_c(df["DayofMonth"])
    X["DayOfWeek"] = _parse_c(df["DayOfWeek"])
    dep = df["DepTime"].astype(float)
    X["DepTime"] = dep
    X["hour"] = dep // 100
    X["minute"] = dep % 100
    X["mins_day"] = dep // 100 * 60 + dep % 100
    X["Distance"] = df["Distance"].astype(float)
    X["log_Distance"] = np.log1p(df["Distance"].astype(float))
    for c in ["UniqueCarrier", "Origin", "Dest"]:
        X[c] = df[c]
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])
    return X


def prepare_train() -> pd.DataFrame:
    return _add_counts(_add_te(prepare(train), train, TE_OOF), train)


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
Xtr_all, ytr_all = prepare_train(), to_y(train)

N_MODELS = 10
DEPTH_CYCLE = [6, 7, 8]
models = []

t0 = time.time()
for k in range(N_MODELS):
    m = xgb.XGBClassifier(
        n_estimators=800,
        max_depth=DEPTH_CYCLE[k % len(DEPTH_CYCLE)],
        learning_rate=0.05,
        tree_method="hist",
        enable_categorical=True,
        min_child_weight=5,
        reg_lambda=2.0,
        subsample=0.8,
        colsample_bytree=0.5 + 0.05 * k,
        random_state=SEED + 100 * k,
        n_jobs=N_JOBS,
    )
    m.fit(Xtr_all, ytr_all)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s for {N_MODELS} models")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = _add_te(prepare(df), df, None)
    X = _add_counts(X, df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
