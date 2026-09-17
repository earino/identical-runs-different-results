"""XGBoost binary classifier for airline delays. Only file the agent edits.

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

# --- fit feature statistics on TRAIN ONLY -------------------------------------
CAT_LEVELS = {
    "Month": pd.Index(sorted(train["Month"].unique())),
    "DayOfWeek": pd.Index(sorted(train["DayOfWeek"].unique())),
    "UniqueCarrier": pd.Index(sorted(train["UniqueCarrier"].unique())),
    "Origin": pd.Index(sorted(train["Origin"].unique())),
    "Dest": pd.Index(sorted(train["Dest"].unique())),
}
DIST_EDGES = np.unique(np.quantile(train["Distance"], np.linspace(0, 1, 11)))[1:-1]

# key columns for target encoding (keys derived from raw df only)
def _te_key(df: pd.DataFrame, name: str) -> pd.Series:
    if name == "Origin":
        return df["Origin"]
    if name == "Dest":
        return df["Dest"]
    if name == "UniqueCarrier":
        return df["UniqueCarrier"]
    if name == "hour":
        return (np.floor(df["DepTime"].astype(float) / 100.0).astype(int) % 24).astype(str)
    if name == "route":
        return df["Origin"] + "_" + df["Dest"]
    if name == "origin_hour":
        return df["Origin"] + "|" + (np.floor(df["DepTime"].astype(float) / 100.0).astype(int) % 24).astype(str)
    if name == "dow_hour":
        return df["DayOfWeek"] + "|" + (np.floor(df["DepTime"].astype(float) / 100.0).astype(int) % 24).astype(str)
    raise KeyError(name)

TE_NAMES = ["Origin", "Dest", "UniqueCarrier", "hour", "route", "origin_hour", "dow_hour"]
TE_M = {"Origin": 100.0, "Dest": 100.0, "UniqueCarrier": 100.0, "hour": 50.0,
        "route": 25.0, "origin_hour": 25.0, "dow_hour": 25.0}

y_all = (train[TARGET] == POSITIVE).astype(int).to_numpy()
GMEAN = y_all.mean()

def _smooth_map(keys: pd.Series, y: np.ndarray, m: float) -> dict:
    df_ = pd.DataFrame({"k": keys.to_numpy(), "y": y})
    g = df_.groupby("k")["y"].agg(["sum", "count"])
    val = (g["sum"] + m * GMEAN) / (g["count"] + m)
    return val.to_dict()

# full-train maps (applied to unseen/holdout rows inside prepare)
FULL_MAPS = {name: _smooth_map(_te_key(train, name), y_all, TE_M[name]) for name in TE_NAMES}

# out-of-fold TE values for the training rows (no leakage into the fit)
N_FOLDS = 5
_oof = {name: np.zeros(len(train)) for name in TE_NAMES}
rng = np.random.RandomState(SEED)
_fold = rng.permutation(len(train)) % N_FOLDS
for f in range(N_FOLDS):
    tr_m = _fold != f
    for name in TE_NAMES:
        mp = _smooth_map(_te_key(train.iloc[np.where(tr_m)[0]], name), y_all[tr_m], TE_M[name])
        keys = _te_key(train, name).to_numpy()[~tr_m]
        _oof[name][~tr_m] = pd.Series(keys).map(mp).fillna(GMEAN).to_numpy()

# log train counts for frequency features
CNT_SPECS = {"Origin": "cnt_origin", "Dest": "cnt_dest", "UniqueCarrier": "cnt_carrier", "route": "cnt_route"}
CNT_MAPS = {c: train[c].value_counts() if c != "route" else (train["Origin"] + "_" + train["Dest"]).value_counts()
            for c in CNT_SPECS}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """ALL feature engineering lives here; predict_proba() calls it on unseen rows."""
    dt = df["DepTime"].astype(float)
    hour = np.floor(dt / 100.0).astype(int) % 24
    minutes = hour * 60 + (dt % 100).astype(int)
    X = pd.DataFrame(index=df.index)
    X["DepTime_min"] = minutes
    X["hour"] = hour
    X["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    X["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    X["Month_num"] = df["Month"].str.replace("c-", "", regex=False).astype(int)
    X["Day_num"] = df["DayofMonth"].str.replace("c-", "", regex=False).astype(int)
    X["Dow_num"] = df["DayOfWeek"].str.replace("c-", "", regex=False).astype(int)
    X["Month_sin"] = np.sin(2 * np.pi * X["Month_num"] / 12.0)
    X["Month_cos"] = np.cos(2 * np.pi * X["Month_num"] / 12.0)
    X["Dow_sin"] = np.sin(2 * np.pi * X["Dow_num"] / 7.0)
    X["Dow_cos"] = np.cos(2 * np.pi * X["Dow_num"] / 7.0)
    X["Distance"] = df["Distance"].astype(float)
    X["Distance_log"] = np.log1p(X["Distance"])
    X["Distance_bin"] = np.searchsorted(DIST_EDGES, X["Distance"], side="right")
    # target encodings (full-train smoothed maps; unseen keys -> global mean)
    for name in TE_NAMES:
        keys = _te_key(df, name)
        X["te_" + name] = keys.map(FULL_MAPS[name]).fillna(GMEAN).to_numpy()
    # frequency features
    X["cnt_origin"] = np.log1p(df["Origin"].map(CNT_MAPS["Origin"]).fillna(0).to_numpy())
    X["cnt_dest"] = np.log1p(df["Dest"].map(CNT_MAPS["Dest"]).fillna(0).to_numpy())
    X["cnt_carrier"] = np.log1p(df["UniqueCarrier"].map(CNT_MAPS["UniqueCarrier"]).fillna(0).to_numpy())
    X["cnt_route"] = np.log1p((df["Origin"] + "_" + df["Dest"]).map(CNT_MAPS["route"]).fillna(0).to_numpy())
    # categoricals
    X["Month"] = pd.Categorical(df["Month"], categories=CAT_LEVELS["Month"])
    X["DayOfWeek"] = pd.Categorical(df["DayOfWeek"], categories=CAT_LEVELS["DayOfWeek"])
    X["UniqueCarrier"] = pd.Categorical(df["UniqueCarrier"], categories=CAT_LEVELS["UniqueCarrier"])
    X["Origin"] = pd.Categorical(df["Origin"], categories=CAT_LEVELS["Origin"])
    X["Dest"] = pd.Categorical(df["Dest"], categories=CAT_LEVELS["Dest"])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
BASE_PARAMS = dict(
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=10.0,
    reg_lambda=1.0,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)
MAX_TREES = 3000
ES_ROUNDS = 50

t0 = time.time()
Xall = prepare(train)
# replace TE columns with out-of-fold encodings for the fit (no leakage)
for name in TE_NAMES:
    Xall["te_" + name] = _oof[name]
yall = to_y(train)
rng = np.random.RandomState(SEED)
idx = rng.permutation(len(Xall))
n_val = int(0.15 * len(Xall))
vi, ti = idx[:n_val], idx[n_val:]

models = []
for seed in (42, 7, 123):
    p = dict(BASE_PARAMS); p["random_state"] = seed
    es_model = xgb.XGBClassifier(n_estimators=MAX_TREES, early_stopping_rounds=ES_ROUNDS, **p)
    es_model.fit(Xall.iloc[ti], yall[ti], eval_set=[(Xall.iloc[vi], yall[vi])], verbose=False)
    best_n = min(int(es_model.best_iteration) + 1, 200)
    m = xgb.XGBClassifier(n_estimators=best_n, **p)
    m.fit(Xall, yall)
    models.append(m)
    print(f"seed {seed}: ES {es_model.best_iteration + 1} -> {best_n} trees")
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    Xp = prepare(df)
    return np.mean([m.predict_proba(Xp)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
