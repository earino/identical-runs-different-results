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
# baseline: treat string columns as categoricals, but drop very high-cardinality ones (names, free text, timestamps)
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}

y_full = (train[TARGET] == POSITIVE).astype(int).to_numpy()
p_bar = float(y_full.mean())

# target encodings, fit on TRAINING DATA ONLY (mapped onto any later dataframe)
def _te_map(keys: pd.DataFrame) -> dict:
    g = pd.DataFrame({"k": keys, "y": y_full}).groupby("k")["y"].agg(["sum", "count"])
    return ((g["sum"] + 20.0 * p_bar) / (g["count"] + 20.0)).to_dict(), g["count"].to_dict()

te_maps, cnt_maps = {}, {}
_y = pd.Series(y_full)
for name, keys in [
    ("Origin", train["Origin"]),
    ("Dest", train["Dest"]),
    ("route", train["Origin"].astype(str) + "_" + train["Dest"].astype(str)),
    ("UniqueCarrier", train["UniqueCarrier"]),
]:
    te_maps[name], cnt_maps[name] = _te_map(keys)

pair_specs = [
    ("car_hour", train["UniqueCarrier"].astype(str) + "_" + (train["DepTime"] // 100).astype(str)),
    ("origin_hour", train["Origin"].astype(str) + "_" + (train["DepTime"] // 100).astype(str)),
]
for name, keys in pair_specs:
    te_maps[name], cnt_maps[name] = _te_map(keys)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    hh = (df["DepTime"].fillna(-1).astype(int) // 100).clip(0, 24)
    X["te_Origin"] = df["Origin"].map(te_maps["Origin"])
    X["te_Dest"] = df["Dest"].map(te_maps["Dest"])
    X["te_route"] = (df["Origin"].astype(str) + "_" + df["Dest"].astype(str)).map(te_maps["route"])
    X["te_carrier"] = df["UniqueCarrier"].map(te_maps["UniqueCarrier"])
    X["te_car_hour"] = (df["UniqueCarrier"].astype(str) + "_" + hh.astype(str)).map(te_maps["car_hour"])
    X["te_org_hour"] = (df["Origin"].astype(str) + "_" + hh.astype(str)).map(te_maps["origin_hour"])
    X["cnt_Origin"] = df["Origin"].map(cnt_maps["Origin"]).astype(float)
    X["cnt_Dest"] = df["Dest"].map(cnt_maps["Dest"]).astype(float)
    X["cnt_route"] = (df["Origin"].astype(str) + "_" + df["Dest"].astype(str)).map(cnt_maps["route"]).astype(float)
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
X = prepare(train)
y = to_y(train)
Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.1, random_state=SEED, stratify=y)
probe = xgb.XGBClassifier(**PARAMS)
probe.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
best_n = int(probe.best_iteration) + 1
print(f"Best iteration: {best_n}")

model = xgb.XGBClassifier(**{**PARAMS, "n_estimators": best_n, "early_stopping_rounds": None})
model.fit(X, y)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
