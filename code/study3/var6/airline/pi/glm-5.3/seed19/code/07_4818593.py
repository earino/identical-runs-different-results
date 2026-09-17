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
    subsample=0.8,
    colsample_bytree=0.8,
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

# traffic-volume counts (fit on train only): hub size, route frequency, airport congestion by hour
_STR_TRAIN = te_struct(train)
CNT_MAPS = {
    "Origin": _STR_TRAIN["Origin"].value_counts(),
    "Route": _STR_TRAIN["Route"].value_counts(),
    "OriginHour": (_STR_TRAIN["Origin"] + "_" + _STR_TRAIN["hour"]).value_counts(),
    "DestHour": (_STR_TRAIN["Dest"] + "_" + _STR_TRAIN["hour"]).value_counts(),
}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Raw dataframe -> feature matrix. Uses only train-fit statistics."""
    X = num_feats(df)
    d = te_struct(df)
    for c in d.columns:
        X["te_" + c] = d[c].map(MAPS_FULL[c]).fillna(GM).to_numpy()
    keys = {
        "Origin": d["Origin"],
        "Route": d["Route"],
        "OriginHour": d["Origin"] + "_" + d["hour"],
        "DestHour": d["Dest"] + "_" + d["hour"],
    }
    for k, s in keys.items():
        X["cnt_" + k] = np.log1p(s.map(CNT_MAPS[k]).fillna(0).astype(float).to_numpy())
    return X


# --- OOF target encoding of train rows ----------------------------------------
d_full = te_struct(train)
CNT_TRAIN = pd.DataFrame({
    "cnt_Origin": np.log1p(d_full["Origin"].map(CNT_MAPS["Origin"]).fillna(0).astype(float).to_numpy()),
    "cnt_Route": np.log1p(d_full["Route"].map(CNT_MAPS["Route"]).fillna(0).astype(float).to_numpy()),
    "cnt_OriginHour": np.log1p((d_full["Origin"] + "_" + d_full["hour"]).map(CNT_MAPS["OriginHour"]).fillna(0).astype(float).to_numpy()),
    "cnt_DestHour": np.log1p((d_full["Dest"] + "_" + d_full["hour"]).map(CNT_MAPS["DestHour"]).fillna(0).astype(float).to_numpy()),
}, index=train.index)
oof_te = pd.DataFrame(index=train.index, columns=d_full.columns, dtype=float)
folds = list(KFold(N_FOLDS, shuffle=True, random_state=SEED).split(train))
for i_tr, i_va in folds:
    m = te_maps(d_full.iloc[i_tr], y_all[i_tr])
    dv = d_full.iloc[i_va]
    for c in oof_te.columns:
        oof_te.iloc[i_va, oof_te.columns.get_loc(c)] = dv[c].map(m[c]).fillna(GM)

X_train_all = pd.concat([num_feats(train), oof_te.rename(columns=lambda c: "te_" + c), CNT_TRAIN], axis=1)

# --- model: pick tree count by mean 5-fold validation curve on train, then refit on all rows
t0 = time.time()
CURVE_TREES = 800
kf5 = KFold(5, shuffle=True, random_state=SEED)
curves = []
for i_tr, i_va in kf5.split(X_train_all):
    m = xgb.XGBClassifier(n_estimators=CURVE_TREES, eval_metric="auc", **PARAMS)
    m.fit(X_train_all.iloc[i_tr], y_all[i_tr], eval_set=[(X_train_all.iloc[i_va], y_all[i_va])], verbose=False)
    r = m.evals_result()["validation_0"]["auc"]
    curves.append(r)
n_cur = min(len(c) for c in curves)
mean_curve = np.mean([c[:n_cur] for c in curves], axis=0)
best_iters = int(np.argmax(mean_curve)) + 1
print(f"tree-count CV: {best_iters} iters (mean 5-fold AUC {mean_curve.max():.4f}), {time.time() - t0:.1f}s")

model = xgb.XGBClassifier(n_estimators=best_iters, **PARAMS)
model.fit(X_train_all, y_all)
print(f"Training time: {time.time() - t0:.1f}s, final: {best_iters} trees")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
