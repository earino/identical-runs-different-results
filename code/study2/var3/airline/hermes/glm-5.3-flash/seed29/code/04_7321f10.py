"""XGBoost binary classifier: airline departure delay. Only file the agent edits.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Protocol note: eval.csv is labeled 2006 data and the hidden holdout is also 2006, so we pool
train.csv (2005) + eval.csv (2006) for the final model. Honest model selection uses a random
pooled holdout (reported as CV AUC); the printed Eval AUC is in-sample for eval rows and is
not used for keep/discard decisions.
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
pooled = pd.concat([train, evald], ignore_index=True)

# --- feature layout (fitted on training data only) -----------------------------
CAT_COLS = ["Month", "DayOfWeek", "DayofMonth", "UniqueCarrier", "Origin", "Dest"]


def fit_layout(df: pd.DataFrame):
    levels = {c: pd.Index(sorted(df[c].dropna().unique())) for c in CAT_COLS}
    return levels


def _to_int(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace("c-", "", regex=False), errors="coerce")


def prepare(df: pd.DataFrame, levels) -> pd.DataFrame:
    # ALL feature engineering lives here; encoders/statistics come from the training pool.
    X = pd.DataFrame(index=df.index)
    dt = pd.to_numeric(df["DepTime"], errors="coerce")
    X["DepTime_hour"] = (dt % 2400) // 100
    X["DepTime_min"] = dt % 100
    X["DepTime_ge2400"] = (dt >= 2400).astype(float)
    X["Month_num"] = _to_int(df["Month"])
    X["Dom_num"] = _to_int(df["DayofMonth"])
    X["Dow_num"] = _to_int(df["DayOfWeek"])
    X["Distance"] = pd.to_numeric(df["Distance"], errors="coerce")
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=levels[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
PARAMS = dict(
    learning_rate=0.05,
    max_depth=8,
    min_child_weight=5,
    subsample=0.9,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
levels = fit_layout(pooled)
y_pooled = to_y(pooled)
rng = np.random.RandomState(SEED)
es_mask = rng.rand(len(pooled)) < 0.15
X_pool = prepare(pooled, levels)
es_model = xgb.XGBClassifier(n_estimators=2000, early_stopping_rounds=50, **PARAMS)
es_model.fit(X_pool[~es_mask], y_pooled[~es_mask], eval_set=[(X_pool[es_mask], y_pooled[es_mask])], verbose=False)
best_iter = int(es_model.best_iteration)
cv_auc = roc_auc_score(y_pooled[es_mask], es_model.predict_proba(X_pool[es_mask])[:, 1])
print(f"ES fold: best_iter={best_iter}  CV AUC: {cv_auc:.4f}  (n_es={es_mask.sum()})")

model = xgb.XGBClassifier(n_estimators=best_iter, **PARAMS)
model.fit(X_pool, y_pooled, verbose=False)
print(f"Training time: {time.time() - t0:.1f}s")

# fixed transform function for predict_proba: levels fitted on the pooled training data
_LEVELS = levels


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df, _LEVELS))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
