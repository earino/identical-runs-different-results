"""XGBoost binary classifier for airline departure delay.

Approach: numeric time features + smoothed target encodings (TE) of categorical
structures incl. Origin/Dest/Carrier x hour interactions. TE maps are fit on
train only; train rows get out-of-fold (OOF) TE values. An early-stopping pass
on a held-out fold picks the tree count, then the final model refits on all rows.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. Module-level `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).
     All feature engineering lives inside prepare(), using only train-fit statistics.
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
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42

K_SMOOTH = 30.0   # target-encoding smoothing count
N_FOLDS = 10      # OOF folds for TE on train rows
PARAMS = dict(
    max_depth=8,
    learning_rate=0.05,
    min_child_weight=5,
    reg_lambda=1.0,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")
y_all = (train[TARGET] == POSITIVE).astype(int).to_numpy()
GM = float(y_all.mean())


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- feature engineering -------------------------------------------------------
def te_struct(df: pd.DataFrame) -> pd.DataFrame:
    """Categorical structures to target-encode."""
    d = pd.DataFrame(index=df.index)
    d["Origin"] = df["Origin"].astype(str)
    d["Dest"] = df["Dest"].astype(str)
    d["UniqueCarrier"] = df["UniqueCarrier"].astype(str)
    d["hour"] = (df["DepTime"].astype(int) // 100).astype(str)
    d["dow"] = df["DayOfWeek"].astype(str)
    d["dom"] = df["DayofMonth"].astype(str)
    d["month"] = df["Month"].astype(str)
    d["Route"] = d["Origin"] + "_" + d["Dest"]
    d["ORoute"] = d["Origin"] + "_" + d["hour"]
    d["DRoute"] = d["Dest"] + "_" + d["hour"]
    d["CHour"] = d["UniqueCarrier"] + "_" + d["hour"]
    d["RHour"] = d["Route"] + "_" + d["hour"]
    d["OCarrier"] = d["Origin"] + "_" + d["UniqueCarrier"]
    d["OHourBin"] = d["Origin"] + "_" + pd.cut(df["DepTime"].astype(int) // 100, [-1, 6, 12, 18, 26]).astype(str)
    return d


def te_maps(d: pd.DataFrame, y: np.ndarray) -> dict:
    """Smoothed target means per level; fit on the given (train) rows only."""
    y = pd.Series(np.asarray(y, dtype=float), index=d.index)
    maps = {}
    for c in d.columns:
        g = y.groupby(d[c]).agg(["mean", "count"])
        maps[c] = (g["count"] * g["mean"] + K_SMOOTH * GM) / (g["count"] + K_SMOOTH)
    return maps


def num_feats(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    dep = df["DepTime"].astype(int)
    tod = dep // 100 * 60 + dep % 100
    X["DepTime"] = dep
    X["tod"] = tod
    X["sin_tod"] = np.sin(2 * np.pi * tod / 1440.0)
    X["cos_tod"] = np.cos(2 * np.pi * tod / 1440.0)
    dist = df["Distance"].astype(float)
    X["Distance"] = dist
    X["log_dist"] = np.log1p(dist)
    return X


MAPS_FULL = te_maps(te_struct(train), y_all)  # train-only fit


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Raw dataframe -> feature matrix. Uses only train-fit statistics."""
    X = num_feats(df)
    d = te_struct(df)
    for c in d.columns:
        X["te_" + c] = d[c].map(MAPS_FULL[c]).fillna(GM).to_numpy()
    return X


# --- OOF target encoding of train rows ----------------------------------------
d_full = te_struct(train)
oof_te = pd.DataFrame(index=train.index, columns=d_full.columns, dtype=float)
folds = list(KFold(N_FOLDS, shuffle=True, random_state=SEED).split(train))
for i_tr, i_va in folds:
    m = te_maps(d_full.iloc[i_tr], y_all[i_tr])
    dv = d_full.iloc[i_va]
    for c in oof_te.columns:
        oof_te.iloc[i_va, oof_te.columns.get_loc(c)] = dv[c].map(m[c]).fillna(GM)

X_train_all = pd.concat([num_feats(train), oof_te.rename(columns=lambda c: "te_" + c)], axis=1)

# --- model: early stopping on held-out fold 0, then refit on all rows ---------
t0 = time.time()
i_tr, i_va = folds[0]
es_model = xgb.XGBClassifier(n_estimators=3000, early_stopping_rounds=60, **PARAMS)
es_model.fit(X_train_all.iloc[i_tr], y_all[i_tr], eval_set=[(X_train_all.iloc[i_va], y_all[i_va])], verbose=False)
best_iters = es_model.best_iteration + 1
print(f"ES pass: {best_iters} iters, {time.time() - t0:.1f}s")

model = xgb.XGBClassifier(n_estimators=best_iters, **PARAMS)
model.fit(X_train_all, y_all)
print(f"Training time: {time.time() - t0:.1f}s, final: {best_iters} trees")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
