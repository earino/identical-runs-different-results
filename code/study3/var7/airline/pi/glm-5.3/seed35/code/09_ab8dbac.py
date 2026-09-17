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
    if name == "te_route15":
        return df["Origin"] + "-" + df["Dest"] + "_" + _bucket(df, 15).astype(str)
    if name == "te_origin30":
        return df["Origin"] + "_" + _bucket(df, 30).astype(str)
    if name == "te_dest30":
        return df["Dest"] + "_" + _bucket(df, 30).astype(str)
    cr = df["UniqueCarrier"] + "-" + df["Origin"] + "-" + df["Dest"]
    if name == "te_carroutehour":
        return cr + "_" + (df["DepTime"] // 100).astype(str)
    if name == "te_carroute30":
        return cr + "_" + _bucket(df, 30).astype(str)
    raise ValueError(name)


TE_M = {"te_orighour": 200, "te_desthour": 200, "te_carrierhour": 300, "te_routehour": 500, "te_route30": 500,
        "te_route15": 800, "te_origin30": 400, "te_dest30": 400, "te_carroutehour": 1000, "te_carroute30": 1500}

# traffic-volume features (train-only counts; volume correlates with congestion)
_oh_cnt = _keys(train, "te_orighour").value_counts()  # origin x hour traffic
_rh_cnt = _keys(train, "te_route30").value_counts()   # route x 30min traffic
_dest_cnt = train["Dest"].value_counts()
_orig_cnt = train["Origin"].value_counts()


def _add_volume(X: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    X["oh_cnt"] = np.log1p(_keys(df, "te_orighour").map(_oh_cnt).fillna(0).to_numpy())
    X["rh_cnt"] = np.log1p(_keys(df, "te_route30").map(_rh_cnt).fillna(0).to_numpy())
    X["dest_cnt"] = np.log1p(df["Dest"].map(_dest_cnt).fillna(0).to_numpy())
    X["orig_cnt"] = np.log1p(df["Origin"].map(_orig_cnt).fillna(0).to_numpy())
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
    # TE difference features: deviations a tree cannot compute in one split
    X["d_ro"] = X["te_routehour"] - X["te_orighour"]
    X["d_r30"] = X["te_route30"] - X["te_routehour"]
    X["d_cr"] = X["te_carroutehour"] - X["te_routehour"]
    X["d_cr30"] = X["te_carroute30"] - X["te_carroutehour"]
    X["d_do"] = X["te_desthour"] - X["te_orighour"]
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model: two-family bagged ensemble ------------------------------------------------
# family 1: full feature set; family 2: stable features only (more robust to TE drift)
X_train = prepare(train)
X_eval = prepare(evald)
_TE_COLS = list(TE_M) + ["d_ro", "d_r30", "d_cr", "d_cr30", "d_do"]
STABLE_COLS = [c for c in X_train.columns if c not in _TE_COLS]


def _mk(seed: int) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=200,
        grow_policy="lossguide",
        max_leaves=64,
        max_depth=0,
        learning_rate=0.06,
        subsample=0.8,
        colsample_bytree=0.4,
        reg_lambda=20,
        tree_method="hist",
        enable_categorical=True,
        random_state=seed,
        n_jobs=N_JOBS,
    )


MODELS = []  # (feature columns, fitted model)
t0 = time.time()
for s in [11, 22, 33, 44, 55, 66]:  # full-feature family
    m = _mk(s)
    m.fit(X_train, y_train)
    MODELS.append((list(X_train.columns), m))
for s in [77, 88, 99]:  # stable-only family
    m = _mk(s)
    m.fit(X_train[STABLE_COLS], y_train)
    MODELS.append((STABLE_COLS, m))
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X[cols])[:, 1] for cols, m in MODELS], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
