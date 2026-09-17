"""XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

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
CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}
_tr_hour = (pd.to_numeric(train["DepTime"], errors="coerce") // 100).clip(0, 25)
car_hour_levels = pd.Index(
    sorted((train["UniqueCarrier"].astype(str) + "_" + _tr_hour.astype(str)).dropna().unique())
)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[CAT_COLS + ["DepTime", "Distance"]].copy()
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    dt = pd.to_numeric(df["DepTime"], errors="coerce")
    hour = (dt // 100).clip(0, 25).astype(float)
    minute = (dt % 100).astype(float)
    X["dep_hour"] = hour
    X["dep_tod"] = np.sin(2 * np.pi * (hour * 60 + minute) / 1440.0)
    X["night"] = ((hour >= 21) | (hour < 5)).astype(float)
    dist = pd.to_numeric(df["Distance"], errors="coerce")
    X["log_dist"] = np.log1p(dist.clip(lower=0))
    X["car_hour"] = pd.Categorical(
        df["UniqueCarrier"].astype(str) + "_" + hour.astype(str), categories=car_hour_levels
    )
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model: small bagged ensemble ---------------------------------------------
def make_model(seed: int) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=60,
        max_depth=6,
        learning_rate=0.06,
        subsample=0.7,
        colsample_bytree=0.7,
        colsample_bynode=0.7,
        tree_method="hist",
        enable_categorical=True,
        random_state=seed,
        n_jobs=N_JOBS,
    )


t0 = time.time()
X_all = prepare(train)
y_all = to_y(train)
models = []
SUBS = (0.6, 0.7, 0.8)
COLS = (0.6, 0.7, 0.8)
LRS = (0.08, 0.1, 0.12)
N_MEMBERS = 20


def build_member(i, seed):
    m = make_model(seed)
    m.set_params(
        max_depth=(4, 6, 8)[i % 3],
        subsample=SUBS[i % 3],
        colsample_bytree=COLS[(i // 3) % 3],
        learning_rate=LRS[(i // 3) % 3],
    )
    return m


# out-of-fold AUC per member -> mild quality weights
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score as _auc

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof_auc = np.zeros(N_MEMBERS)
for f, (itr, iva) in enumerate(skf.split(X_all, y_all)):
    Xtr, Xva = X_all.iloc[itr], X_all.iloc[iva]
    ytr, yva = y_all[itr], y_all[iva]
    for i in range(N_MEMBERS):
        m = build_member(i, SEED + 1000 + i * 101)
        m.fit(Xtr, ytr)
        oof_auc[i] += _auc(yva, m.predict_proba(Xva)[:, 1]) / skf.n_splits
print("OOF member AUCs:", np.round(oof_auc, 4))

w = np.clip(oof_auc - 0.65, 0.0, None) ** 8
if w.sum() <= 0:
    w = np.ones(N_MEMBERS)
w = 0.35 + 0.65 * w / w.max()  # sharper but still bounded tilt
w = w / w.sum()
print("weights:", np.round(w, 4))

for i in range(N_MEMBERS):
    m = build_member(i, SEED + i * 101)
    m.fit(X_all, y_all)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    P = prepare(df)
    preds = np.stack([m.predict_proba(P)[:, 1] for m in models])
    return (w[:, None] * preds).sum(axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
