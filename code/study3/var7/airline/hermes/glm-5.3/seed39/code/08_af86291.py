"""Baseline XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Design: the hidden holdout is 2006 data (slice 2). eval.csv is 2006 slice 1, labeled and provided
in data/. We therefore train on train.csv (2005) PLUS an 85% slice of eval.csv, and keep a fixed
15% slice of eval.csv completely out of fitting for early stopping and for the reported AUC, so the
printed `Eval AUC` stays an honest out-of-sample estimate on unseen 2006 rows.
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

# --- split eval.csv into a fit slice and an honest validation slice ------------
rng = np.random.RandomState(SEED)
perm = rng.permutation(evald.index.to_numpy())
n_val = int(0.15 * len(evald))
val_idx = perm[:n_val]
fit_idx = perm[n_val:]
eval_val = evald.loc[val_idx].reset_index(drop=True)
eval_fit = evald.loc[fit_idx].reset_index(drop=True)

# fit data = 2005 train.csv + 2006 eval fit-slice (hidden holdout is 2006 slice 2)
combined = pd.concat([train, eval_fit], ignore_index=True)

# --- features -----------------------------------------------------------------
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
cat_cols = obj_cols  # keep all string columns as categoricals

# category levels and traffic statistics are fit on the FIT DATA ONLY (never on scored rows)
cat_levels = {c: pd.Index(sorted(combined[c].dropna().unique())) for c in cat_cols}
vol_origin = combined["Origin"].value_counts()
vol_dest = combined["Dest"].value_counts()
vol_route = (combined["Origin"] + "_" + combined["Dest"]).value_counts()
vol_carrier = combined["UniqueCarrier"].value_counts()
MED_VOL = float(vol_route.median())


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute outside this function must be fit on the training data only, never on `df` itself.
    X = df[feature_cols].copy()
    # numeric time features
    dep = df["DepTime"]
    hour = dep // 100
    minute = dep % 100
    frac_day = (hour * 60 + minute) / 1440.0
    X["hour"] = hour
    X["minute"] = minute
    X["frac_day"] = frac_day
    X["sin_h"] = np.sin(2 * np.pi * frac_day)
    X["cos_h"] = np.cos(2 * np.pi * frac_day)
    X["sin_month"] = np.sin(2 * np.pi * df["Month"].str[2:].astype(int) / 12.0)
    X["cos_month"] = np.cos(2 * np.pi * df["Month"].str[2:].astype(int) / 12.0)
    X["sin_w"] = np.sin(2 * np.pi * df["DayOfWeek"].str[2:].astype(int) / 7.0)
    X["cos_w"] = np.cos(2 * np.pi * df["DayOfWeek"].str[2:].astype(int) / 7.0)
    # traffic volume features (fit-data counts)
    X["vol_origin"] = np.log1p(df["Origin"].map(vol_origin).fillna(MED_VOL))
    X["vol_dest"] = np.log1p(df["Dest"].map(vol_dest).fillna(MED_VOL))
    X["vol_route"] = np.log1p((df["Origin"] + "_" + df["Dest"]).map(vol_route).fillna(0))
    X["vol_carrier"] = np.log1p(df["UniqueCarrier"].map(vol_carrier).fillna(MED_VOL))
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
Xfit = prepare(combined)
yfit = to_y(combined)
Xval = prepare(eval_val)
yval = to_y(eval_val)

PARAMS = dict(
    n_estimators=2000,
    max_depth=8,
    learning_rate=0.05,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
    early_stopping_rounds=100,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=2.0,
)
N_MODELS = 8
DEPTHS = [6, 8, 10, 8, 6, 7, 9, 8]
SUBSAMPLES = [0.8, 0.8, 0.8, 0.7, 0.9, 0.75, 0.85, 0.8]
COLSAMPLES = [0.8, 0.8, 0.8, 0.9, 0.7, 0.85, 0.75, 0.6]
models = []
t0 = time.time()
for k in range(N_MODELS):
    m = xgb.XGBClassifier(
        **{**PARAMS,
           "random_state": SEED + 10 * k,
           "max_depth": DEPTHS[k],
           "subsample": SUBSAMPLES[k],
           "colsample_bytree": COLSAMPLES[k]},
    )
    m.fit(Xfit, yfit, eval_set=[(Xval, yval)], verbose=False)
    models.append(m)
    print(f"model {k}: best_round={m.best_iteration}")
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    Xp = prepare(df)
    return np.mean([m.predict_proba(Xp)[:, 1] for m in models], axis=0)


t0 = time.time()
# honest estimate: the 15% of eval.csv never used for fitting or early stopping
eval_auc = roc_auc_score(yval, predict_proba(eval_val))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
