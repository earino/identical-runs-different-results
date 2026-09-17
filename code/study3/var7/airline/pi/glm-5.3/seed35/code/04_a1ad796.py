"""XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Method: stable numeric features (time-of-day, calendar) + native categoricals for carrier/origin/dest,
plus out-of-fold target-encoded interaction features (origin x hour, dest x hour, carrier x hour).
OOF encoding keeps the TE features honest for training rows so the GBM doesn't over-trust them.
Bagged ensemble of XGB seeds for variance reduction (2005 -> 2006 year shift punishes variance hard).
"""
import json
import os
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

TASK = json.load(open("task.json"))
TARGET = TASK["target"]
POSITIVE = TASK["positive_label"]
ID_COLS = TASK.get("id_columns", [])
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- target encoding maps (fit on training data ONLY) --------------------------
y_train = (train[TARGET] == POSITIVE).astype(int).to_numpy()
y_train_f = y_train.astype(float)
HOUR = train["DepTime"] // 100


def _bucket(df: pd.DataFrame, nmin: int) -> pd.Series:
    return (df["DepTime"] // 100) * (60 // nmin) + (df["DepTime"] % 100) // nmin


def _keys(df: pd.DataFrame, name: str) -> pd.Series:
    h = df["DepTime"] // 100
    if name == "te_orighour":
        return df["Origin"] + "_" + h.astype(str)
    if name == "te_desthour":
        return df["Dest"] + "_" + h.astype(str)
    if name == "te_carrierhour":
        return df["UniqueCarrier"] + "_" + h.astype(str)
    if name == "te_routehour":
        return df["Origin"] + "-" + df["Dest"] + "_" + h.astype(str)
    if name == "te_route30":
        return df["Origin"] + "-" + df["Dest"] + "_" + _bucket(df, 30).astype(str)
    raise ValueError(name)


TE_M = {"te_orighour": 200, "te_desthour": 200, "te_carrierhour": 300, "te_routehour": 500, "te_route30": 500}

# traffic-volume features (train-only counts; volume correlates with congestion)
_oh_cnt = _keys(train, "te_orighour").value_counts()  # origin x hour traffic
_rh_cnt = _keys(train, "te_route30").value_counts()   # route x 30min traffic


def _add_volume(X: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    X["oh_cnt"] = np.log1p(_keys(df, "te_orighour").map(_oh_cnt).fillna(0).to_numpy())
    X["rh_cnt"] = np.log1p(_keys(df, "te_route30").map(_rh_cnt).fillna(0).to_numpy())
    return X


def _te_map(keys: pd.Series, y: np.ndarray, m: int) -> pd.Series:
    g = pd.DataFrame({"k": keys, "y": y}).groupby("k")["y"].agg(["sum", "count"])
    return (g["sum"] + 0.5 * m) / (g["count"] + m)


# out-of-fold values for training rows (honest: no row sees its own label)
_oof = {k: pd.Series(np.nan, index=train.index) for k in TE_M}
for i_tr, i_va in KFold(5, shuffle=True, random_state=0).split(train):
    for k, m in TE_M.items():
        _oof[k].iloc[i_va] = _te_map(_keys(train.iloc[i_tr], k), y_train_f[i_tr], m).reindex(
            _keys(train.iloc[i_va], k)).values
# full-sample maps for any unseen dataframe (train-only statistics)
_full = {k: _te_map(_keys(train, k), y_train_f, m) for k, m in TE_M.items()}

# --- features -----------------------------------------------------------------
_carr_lv = pd.Index(sorted(train["UniqueCarrier"].unique()))
_orig_lv = pd.Index(sorted(train["Origin"].unique()))
_dest_lv = pd.Index(sorted(train["Dest"].unique()))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    dep = df["DepTime"].astype(int)
    X["month"] = df["Month"].str.replace("c-", "", regex=False).astype(int)
    X["day"] = df["DayofMonth"].str.replace("c-", "", regex=False).astype(int)
    X["dow"] = df["DayOfWeek"].str.replace("c-", "", regex=False).astype(int)
    X["dep_hour"] = dep // 100
    X["dep_min"] = dep % 100
    X["dep_ssm"] = X["dep_hour"] * 60 + X["dep_min"]
    X["distance"] = df["Distance"].astype(float)
    X["UniqueCarrier"] = pd.Categorical(df["UniqueCarrier"], categories=_carr_lv)
    X["Origin"] = pd.Categorical(df["Origin"], categories=_orig_lv)
    X["Dest"] = pd.Categorical(df["Dest"], categories=_dest_lv)
    X = _add_volume(X, df)
    if df is train:  # training matrix: honest OOF values
        for k in TE_M:
            X[k] = _oof[k].fillna(0.5).values
    else:  # unseen rows: full-train maps
        for k in TE_M:
            X[k] = _keys(df, k).map(_full[k]).fillna(0.5).values
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model: bagged XGB seeds ---------------------------------------------------
X_train = prepare(train)
X_eval = prepare(evald)
MODELS = []
t0 = time.time()
for s in [11, 22, 33, 44, 55, 66, 77, 88, 99]:
    m = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.7,
        colsample_bytree=0.6,
        tree_method="hist",
        enable_categorical=True,
        random_state=s,
        n_jobs=N_JOBS,
    )
    m.fit(X_train, y_train)
    MODELS.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in MODELS], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
