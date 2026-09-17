"""XGBoost binary classifier for airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

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
from sklearn.model_selection import KFold

TASK = json.load(open("task.json"))
TARGET = TASK["target"]
POSITIVE = TASK["positive_label"]
ID_COLS = TASK.get("id_columns", [])
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")
y_train = (train[TARGET] == POSITIVE).astype(int).to_numpy()
PRIOR = float(y_train.mean())

# --- categorical levels (fit on train only) --------------------------------------
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}
route_levels = pd.Index(sorted((train["Origin"] + "-" + train["Dest"]).unique()))

# --- target-encoding keys --------------------------------------------------------
TE_KEYS = ["carrier", "origin", "dest", "route", "carrier_hour", "origin_hour", "dest_hour"]
SMOOTH = 25


def _keys(df: pd.DataFrame) -> pd.DataFrame:
    hour = (df["DepTime"] // 100).astype(int)
    k = pd.DataFrame(index=df.index)
    k["carrier"] = df["UniqueCarrier"].to_numpy()
    k["origin"] = df["Origin"].to_numpy()
    k["dest"] = df["Dest"].to_numpy()
    k["route"] = (df["Origin"] + "-" + df["Dest"]).to_numpy()
    k["carrier_hour"] = (df["UniqueCarrier"] + "_" + hour.astype(str)).to_numpy()
    k["origin_hour"] = (df["Origin"] + "_" + hour.astype(str)).to_numpy()
    k["dest_hour"] = (df["Dest"] + "_" + hour.astype(str)).to_numpy()
    return k


def _te_fit(keys: np.ndarray, y: np.ndarray) -> pd.Series:
    agg = pd.DataFrame({"k": keys, "y": y}).groupby("k")["y"].agg(["sum", "count"])
    return (agg["sum"] + SMOOTH * PRIOR) / (agg["count"] + SMOOTH)


k_train = _keys(train)
te_map = {c: _te_fit(k_train[c].to_numpy(), y_train) for c in TE_KEYS}

# out-of-fold TE values for the training rows (avoids leakage into the fit)
oof = np.zeros((len(train), len(TE_KEYS)))
fold = np.zeros(len(train), dtype=int)
for f, (_, va) in enumerate(KFold(5, shuffle=True, random_state=SEED).split(np.zeros(len(train)))):
    fold[va] = f
for j, c in enumerate(TE_KEYS):
    keys = k_train[c].to_numpy()
    oof[:, j] = PRIOR
    for f in range(5):
        m = (fold == f)
        s = _te_fit(keys[~m], y_train[~m])
        oof[m, j] = pd.Series(keys[m]).map(s).fillna(PRIOR).to_numpy()
oof = pd.DataFrame(oof, columns=["te_" + c for c in TE_KEYS], index=train.index)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """ALL feature engineering lives here: predict_proba() calls this on unseen rows."""
    X = pd.DataFrame(index=df.index)
    mon = df["Month"].str.replace("c-", "", regex=False).astype(int)
    dom = df["DayofMonth"].str.replace("c-", "", regex=False).astype(int)
    dow = df["DayOfWeek"].str.replace("c-", "", regex=False).astype(int)
    X["month"] = mon
    X["day"] = dom
    X["dow"] = dow
    mins = (df["DepTime"].astype(int) // 100) * 60 + (df["DepTime"].astype(int) % 100)
    X["dep_minutes"] = mins
    X["hour"] = mins // 60
    X["sin_dep"] = np.sin(2 * np.pi * mins / 1440.0)
    X["cos_dep"] = np.cos(2 * np.pi * mins / 1440.0)
    doy = (mon - 1) * 31 + dom
    X["doy"] = doy
    X["sin_doy"] = np.sin(2 * np.pi * doy / 365.0)
    X["cos_doy"] = np.cos(2 * np.pi * doy / 365.0)
    X["sin_dow"] = np.sin(2 * np.pi * dow / 7.0)
    X["dist"] = df["Distance"].astype(float)
    X["log_dist"] = np.log1p(df["Distance"].astype(float))
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    X["route"] = pd.Categorical(df["Origin"] + "-" + df["Dest"], categories=route_levels)
    k = _keys(df)
    for c in TE_KEYS:
        X["te_" + c] = pd.Series(k[c].to_numpy()).map(te_map[c]).fillna(PRIOR).to_numpy()
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model ----------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=400,
    max_depth=7,
    learning_rate=0.05,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
X_train = prepare(train)
for c in TE_KEYS:  # replace in-sample TE with the OOF version for training rows
    X_train["te_" + c] = oof["te_" + c].to_numpy()
model.fit(X_train, y_train)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
