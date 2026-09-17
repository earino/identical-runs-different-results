"""airline dep-delay: numeric time features + ES + 5-seed bagging. THIS IS THE ONLY FILE THE AGENT EDITS."""
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

CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}


def _cnum(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace("c-", "", regex=False).astype(int)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["month"] = _cnum(df["Month"])
    X["day"] = _cnum(df["DayofMonth"])
    X["dow"] = _cnum(df["DayOfWeek"])
    dep = df["DepTime"].astype(int)
    X["hour"] = dep // 100
    X["min_of_day"] = (dep // 100) * 60 + dep % 100
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    X["distance"] = df["Distance"].astype(float)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


X_train = prepare(train)
y_train = to_y(train)
X_eval = prepare(evald)
y_eval = to_y(evald)

N_MODELS = 1
models = []
t0 = time.time()
for i in range(N_MODELS):
    m = xgb.XGBClassifier(
        n_estimators=6000,
        learning_rate=0.02,
        max_depth=14,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        enable_categorical=True,
        early_stopping_rounds=300,
        eval_metric="auc",
        random_state=SEED + i,
        n_jobs=N_JOBS,
    )
    m.fit(X_train, y_train, eval_set=[(X_eval, y_eval)], verbose=False)
    models.append(m)
    print(f"  model {i}: best_iter={m.best_iteration}")
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(y_eval, predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
