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

TASK = json.load(open("task.json"))
TARGET = TASK["target"]
POSITIVE = TASK["positive_label"]
ID_COLS = TASK.get("id_columns", [])
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- features -----------------------------------------------------------------
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]


def _strip_c(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace("c-", "", regex=False).astype(int)


# categorical levels learned from the training data only
route_levels = pd.Index(sorted((train["Origin"] + "_" + train["Dest"]).unique()))
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    # calendar numerics
    X["month"] = _strip_c(df["Month"])
    X["day"] = _strip_c(df["DayofMonth"])
    X["dow"] = _strip_c(df["DayOfWeek"])
    # departure time
    dep = df["DepTime"].astype(int)
    hour = (dep // 100).clip(0, 24)
    minute = dep % 100
    X["hour"] = hour
    X["dep_minutes"] = (hour * 60 + minute).clip(0, 1440)
    X["dep_sin"] = np.sin(2 * np.pi * X["dep_minutes"] / 1440)
    X["dep_cos"] = np.cos(2 * np.pi * X["dep_minutes"] / 1440)
    # cyclic calendar
    X["month_sin"] = np.sin(2 * np.pi * X["month"] / 12)
    X["month_cos"] = np.cos(2 * np.pi * X["month"] / 12)
    X["dow_sin"] = np.sin(2 * np.pi * X["dow"] / 7)
    X["dow_cos"] = np.cos(2 * np.pi * X["dow"] / 7)
    X["day_sin"] = np.sin(2 * np.pi * X["day"] / 31)
    X["day_cos"] = np.cos(2 * np.pi * X["day"] / 31)
    # distance
    X["distance"] = df["Distance"].astype(float)
    X["log_distance"] = np.log1p(X["distance"])
    # categoricals (native)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    route = df["Origin"] + "_" + df["Dest"]
    X["route"] = pd.Categorical(route, categories=route_levels)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
def make_model(**kw) -> xgb.XGBClassifier:
    params = dict(
        n_estimators=2000,
        max_depth=6,
        learning_rate=0.1,
        tree_method="hist",
        enable_categorical=True,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=1.0,
        early_stopping_rounds=50,
        random_state=SEED,
        n_jobs=N_JOBS,
    )
    params.update(kw)
    return xgb.XGBClassifier(**params)


t0 = time.time()
X_all, y_all = prepare(train), to_y(train)
# internal validation split for early stopping (labels of eval.csv are never used for training)
rng = np.random.RandomState(SEED)
val_idx = rng.rand(len(X_all)) < 0.1
X_tr, y_tr = X_all[~val_idx], y_all[~val_idx]
X_val, y_val = X_all[val_idx], y_all[val_idx]

es_model = make_model()
es_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
best_iter = es_model.best_iteration
print(f"Early-stop model: best_iter={best_iter}, val AUC={es_model.best_score:.4f}")

final_model = make_model(n_estimators=best_iter + 1, early_stopping_rounds=None)
final_model.fit(X_all, y_all, verbose=False)
print(f"Training time: {time.time() - t0:.1f}s")

model = final_model


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
