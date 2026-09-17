"""XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Notes: train=2005, eval/holdout=2006 (time shift). Deep/high-capacity models overfit 2005 noise,
shallow (d4) trees with many low-lr rounds generalize best. Native categorical splits on high-cardinality
interactions (route) hurt; a heavily-smoothed target encoding of origin x hour-of-day as a *numeric*
feature transfers across the year boundary and adds ~+0.002 AUC.
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

# --- features -----------------------------------------------------------------
RAW_NUM = ["DepTime", "Distance"]
CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
N_SEEDS = 5

y_train = (train[TARGET] == POSITIVE).astype(int).to_numpy()
prior = float(y_train.mean())

# group keys for target encoding, computed from raw columns of any df
TE_SPEC = [
    # (name, key-function, smoothing m): time-bin x airport target encodings at two granularities
    ("te_o_k25", lambda df: (df["DepTime"] // 25).astype(str) + "_" + df["Origin"].astype(str), 50),
    ("te_d_k25", lambda df: (df["DepTime"] // 25).astype(str) + "_" + df["Dest"].astype(str), 50),
    ("te_o_k12", lambda df: (df["DepTime"] // 12).astype(str) + "_" + df["Origin"].astype(str), 100),
    ("te_d_k12", lambda df: (df["DepTime"] // 12).astype(str) + "_" + df["Dest"].astype(str), 100),
    ("te_c_k12", lambda df: (df["DepTime"] // 12).astype(str) + "_" + df["UniqueCarrier"].astype(str), 100),
]
# target-encoding maps fit on the TRAINING data only (module level, from `train`)
te_maps = {}
for name, keyfn, m in TE_SPEC:
    g = pd.DataFrame({"g": keyfn(train).to_numpy(), "y": y_train})
    agg = g.groupby("g")["y"].agg(["sum", "count"])
    te_maps[name] = ((agg["sum"] + m * prior) / (agg["count"] + m))

# categorical levels fit on TRAIN only
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[RAW_NUM].copy()
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    for name, keyfn, m in TE_SPEC:
        X[name] = keyfn(df).map(te_maps[name]).astype(float).fillna(prior).to_numpy()
    return X


# For the training rows we use out-of-fold TE values (same transform the holdout gets, but without
# leaking each row's own label into its own feature value).
oof_te = {}
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
idx = np.arange(len(train))
for name, keyfn, m in TE_SPEC:
    vals = np.full(len(train), prior)
    keys = keyfn(train)
    for tr_i, va_i in kf.split(idx):
        g = pd.DataFrame({"g": keys.iloc[tr_i].to_numpy(), "y": y_train[tr_i]})
        agg = g.groupby("g")["y"].agg(["sum", "count"])
        sm = (agg["sum"] + m * prior) / (agg["count"] + m)
        vals[va_i] = keys.iloc[va_i].map(sm).astype(float).fillna(prior).to_numpy()
    oof_te[name] = vals


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


X_train_full = prepare(train)  # what eval/holdout rows get (full-train TE)
X_train_fit = X_train_full.copy()
for name, keyfn, m in TE_SPEC:
    X_train_fit[name] = oof_te[name]  # training rows see honest OOF values

# --- model --------------------------------------------------------------------
PARAMS = dict(
    n_estimators=600,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.9,
    colsample_bytree=0.6,
    reg_alpha=0.5,
    tree_method="hist",
    enable_categorical=True,
)

t0 = time.time()
models = []
for s in range(N_SEEDS):
    m = xgb.XGBClassifier(**PARAMS, random_state=SEED + s, n_jobs=N_JOBS)
    m.fit(X_train_fit, y_train)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s ({N_SEEDS} seeds)")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
