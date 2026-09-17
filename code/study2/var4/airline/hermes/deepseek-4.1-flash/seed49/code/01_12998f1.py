"""XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).
     All feature engineering lives in `prepare()`, which predict_proba() calls on unseen rows.
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

# --- feature schema (fitted on training data only) ------------------------------
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}


# --- feature engineering -------------------------------------------------------
def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything
    # computed outside this function (other than training set statistics) will NOT reach the hidden holdout.
    X = df[feature_cols].copy()

    # scheduled departure time: hhmm integer -> clock features (values >= 2400 mean past midnight)
    dep = X["DepTime"].to_numpy(dtype=np.int64)
    hour = dep // 100
    minute = dep % 100
    X["dep_hour"] = np.where(hour >= 24, hour - 24, hour)
    X["dep_minute"] = minute
    X["dep_tod"] = X["dep_hour"] * 60 + minute           # minutes after midnight
    X["dep_red_eye"] = (hour >= 24).astype(np.int8)

    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


PARAMS = dict(
    n_estimators=3000,
    max_depth=7,
    learning_rate=0.05,
    tree_method="hist",
    enable_categorical=True,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=SEED,
    n_jobs=N_JOBS,
    early_stopping_rounds=50,
)

t0 = time.time()
Xtr = prepare(train)
ytr = to_y(train)
Xa, Xb, ya, yb = train_test_split(Xtr, ytr, test_size=0.1, random_state=SEED, stratify=ytr)
probe = xgb.XGBClassifier(**PARAMS)
probe.fit(Xa, ya, eval_set=[(Xb, yb)], verbose=False)
best_n = int(probe.best_iteration) + 1
print(f"probe best_iteration={best_n} val_logloss={probe.best_score:.4f}")

final_params = {k: v for k, v in PARAMS.items() if k != "early_stopping_rounds"}
final_params["n_estimators"] = best_n
model = xgb.XGBClassifier(**final_params)
model.fit(Xtr, ytr, verbose=False)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
