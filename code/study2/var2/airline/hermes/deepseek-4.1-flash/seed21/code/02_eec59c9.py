"""XGBoost binary classifier for the airline delay task. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).
All feature engineering lives in `prepare()`, which is the single code path used both for fitting and for
scoring unseen rows (including the hidden holdout). Encoders/statistics are fitted on training data only.
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

CAT_FEATS = ["UniqueCarrier", "Origin", "Dest"]
CAT_LEVELS = {c: pd.Index(sorted(train[c].astype(str).unique())) for c in CAT_FEATS}
HOUR_LEVELS = pd.Index([str(i) for i in range(24)])

# --- target encoding of (airport, scheduled hour), fitted on training data only -----------------
# The intraday delay ramp differs by airport and is the one relation that transfers across years;
# giving the model an explicit smoothed rate (rather than making it rediscover it from 300 airports)
# sharpens the splits. Training rows use out-of-fold values so the model cannot memorise itself;
# unseen rows use the full-training-data map.
TE_NAMES = ("o_hour", "d_hour")
TE_SMOOTH = 20.0
TE_NFOLD = 5
PRIOR = float((train[TARGET] == POSITIVE).mean())


def _te_key(df: pd.DataFrame, name: str) -> pd.Series:
    hour = ((df["DepTime"].astype(int) // 100) % 24).astype(str)
    if name == "o_hour":
        return df["Origin"].astype(str) + "_" + hour
    return df["Dest"].astype(str) + "_" + hour


def _fit_te(df: pd.DataFrame, y: np.ndarray) -> dict:
    out = {}
    for name in TE_NAMES:
        grp = pd.DataFrame({"k": _te_key(df, name).to_numpy(), "y": y}).groupby("k")["y"].agg(["sum", "count"])
        out[name] = ((grp["sum"] + PRIOR * TE_SMOOTH) / (grp["count"] + TE_SMOOTH)).to_dict()
    return out


TE_MAP = _fit_te(train, (train[TARGET] == POSITIVE).to_numpy().astype(float))

# out-of-fold encoding for the training rows
__rng = np.random.RandomState(0)
__fold = __rng.randint(0, TE_NFOLD, len(train))
y_all = (train[TARGET] == POSITIVE).to_numpy().astype(float)
OOF_TE = {}
for name in TE_NAMES:
    vals = np.full(len(train), PRIOR)
    k_all = _te_key(train, name).to_numpy()
    for f in range(TE_NFOLD):
        m = __fold != f
        grp = pd.DataFrame({"k": k_all[m], "y": y_all[m]}).groupby("k")["y"].agg(["sum", "count"])
        te = (grp["sum"] + y_all[m].mean() * TE_SMOOTH) / (grp["count"] + TE_SMOOTH)
        vals[~m] = pd.Series(k_all[~m]).map(te).fillna(y_all[m].mean()).to_numpy()
    OOF_TE[name] = vals


def prepare(df: pd.DataFrame, te_override: dict | None = None) -> pd.DataFrame:
    """Raw dataframe -> model matrix. Used for training AND for predict_proba on unseen rows."""
    dt = df["DepTime"].astype(int)
    hour = (dt // 100) % 24
    minute = (dt % 100).clip(0, 59)
    tod = hour + minute / 60.0
    X = pd.DataFrame(index=df.index)
    X["hour"] = hour
    X["minute"] = minute
    X["tod"] = tod
    X["tod_sin"] = np.sin(2 * np.pi * tod / 24)
    X["tod_cos"] = np.cos(2 * np.pi * tod / 24)
    X["Distance"] = df["Distance"].astype(float)
    X["log_dist"] = np.log1p(df["Distance"].astype(float))
    month = df["Month"].astype(str).str.replace("c-", "", regex=False).astype(int)
    dom = df["DayofMonth"].astype(str).str.replace("c-", "", regex=False).astype(int)
    dow = df["DayOfWeek"].astype(str).str.replace("c-", "", regex=False).astype(int)
    X["Month_n"] = month
    X["Day_n"] = dom
    X["Dow_n"] = dow
    X["is_weekend"] = (dow >= 6).astype(int)
    X["hour_cat"] = pd.Categorical(hour.astype(str), categories=HOUR_LEVELS)
    for c in CAT_FEATS:
        X[c] = pd.Categorical(df[c].astype(str), categories=CAT_LEVELS[c])
    for name in TE_NAMES:
        if te_override is not None and te_override.get(name) is not None:
            X["te_" + name] = te_override[name]
        else:
            X["te_" + name] = _te_key(df, name).map(TE_MAP[name]).fillna(PRIOR).to_numpy()
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=600,
    max_depth=3,
    learning_rate=0.05,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(prepare(train, te_override=OOF_TE), to_y(train))
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
