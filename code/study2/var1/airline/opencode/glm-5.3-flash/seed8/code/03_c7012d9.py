"""XGBoost binary classifier on airline delays. THIS IS THE ONLY FILE THE AGENT EDITS.

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
from sklearn.model_selection import train_test_split

TASK = json.load(open("task.json"))
TARGET = TASK["target"]
POSITIVE = TASK["positive_label"]
ID_COLS = TASK.get("id_columns", [])
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- features -----------------------------------------------------------------
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}

SMOOTH = 20.0


def te_fit(df: pd.DataFrame, y: np.ndarray):
    """Fit smoothed target encodings + counts on (df, y) only."""
    p_bar = float(y.mean())
    specs = {
        "Origin": df["Origin"],
        "Dest": df["Dest"],
        "route": df["Origin"].astype(str) + "_" + df["Dest"].astype(str),
        "UniqueCarrier": df["UniqueCarrier"],
        "car_hour": df["UniqueCarrier"].astype(str) + "_" + (df["DepTime"].fillna(-1).astype(int) // 100).astype(str),
        "org_hour": df["Origin"].astype(str) + "_" + (df["DepTime"].fillna(-1).astype(int) // 100).astype(str),
    }
    te, cnt = {}, {}
    for name, keys in specs.items():
        g = pd.DataFrame({"k": keys.to_numpy(), "y": y}).groupby("k")["y"].agg(["sum", "count"])
        te[name] = ((g["sum"] + SMOOTH * p_bar) / (g["count"] + SMOOTH)).to_dict()
        cnt[name] = g["count"].to_dict()
    return te, cnt


def prepare(df: pd.DataFrame, te=None, cnt=None) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    hh = (df["DepTime"].fillna(-1).astype(int) // 100).clip(0, 24)
    route = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    X["te_Origin"] = df["Origin"].map(te["Origin"])
    X["te_Dest"] = df["Dest"].map(te["Dest"])
    X["te_route"] = route.map(te["route"])
    X["te_carrier"] = df["UniqueCarrier"].map(te["UniqueCarrier"])
    X["te_car_hour"] = (df["UniqueCarrier"].astype(str) + "_" + hh.astype(str)).map(te["car_hour"])
    X["te_org_hour"] = (df["Origin"].astype(str) + "_" + hh.astype(str)).map(te["org_hour"])
    X["cnt_Origin"] = df["Origin"].map(cnt["Origin"]).astype(float)
    X["cnt_Dest"] = df["Dest"].map(cnt["Dest"]).astype(float)
    X["cnt_route"] = route.map(cnt["route"]).astype(float)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
PARAMS = dict(
    n_estimators=3000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=10.0,
    reg_lambda=1.0,
    tree_method="hist",
    enable_categorical=True,
    eval_metric="auc",
    early_stopping_rounds=100,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
y = to_y(train)
tr_idx, va_idx = train_test_split(np.arange(len(train)), test_size=0.1, random_state=SEED, stratify=y)
te_tr, cnt_tr = te_fit(train.iloc[tr_idx], y[tr_idx])
Xtr = prepare(train.iloc[tr_idx], te_tr, cnt_tr)
Xva = prepare(train.iloc[va_idx], te_tr, cnt_tr)  # encodings fit on tr rows only: no leakage
probe = xgb.XGBClassifier(**PARAMS)
probe.fit(Xtr, y[tr_idx], eval_set=[(Xva, y[va_idx])], verbose=False)
best_n = int(probe.best_iteration) + 1
print(f"Best iteration: {best_n}")

te_full, cnt_full = te_fit(train, y)
model = xgb.XGBClassifier(**{**PARAMS, "n_estimators": best_n, "early_stopping_rounds": None})
model.fit(prepare(train, te_full, cnt_full), y)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df, te_full, cnt_full))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
