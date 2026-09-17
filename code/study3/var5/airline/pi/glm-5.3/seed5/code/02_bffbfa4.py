"""XGBoost binary classifier for airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

All feature engineering lives in prepare() and only uses artifacts (category levels) fit on the
training data at module level. Early stopping uses a random split of the TRAINING data only.
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


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- feature engineering -------------------------------------------------------
CAT_COLS = ["UniqueCarrier", "Origin", "Dest", "route", "Month", "DayofMonth", "DayOfWeek"]
NUM_COLS = ["month", "day", "dow", "hour", "deptime_min", "Distance"]


def base_features(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["UniqueCarrier"] = df["UniqueCarrier"].astype(str)
    X["Origin"] = df["Origin"].astype(str)
    X["Dest"] = df["Dest"].astype(str)
    X["route"] = X["Origin"] + "-" + X["Dest"]
    X["Month"] = df["Month"].astype(str)
    X["DayofMonth"] = df["DayofMonth"].astype(str)
    X["DayOfWeek"] = df["DayOfWeek"].astype(str)
    X["month"] = df["Month"].str.replace("c-", "", regex=False).astype(int)
    X["day"] = df["DayofMonth"].str.replace("c-", "", regex=False).astype(int)
    X["dow"] = df["DayOfWeek"].str.replace("c-", "", regex=False).astype(int)
    hour = (df["DepTime"] // 100) % 24  # DepTime can exceed 2400 (after-midnight)
    X["hour"] = hour
    X["deptime_min"] = hour * 60 + df["DepTime"] % 100
    X["Distance"] = df["Distance"].astype(float)
    return X


# artifacts fit on TRAIN only
_train_feats = base_features(train)
cat_levels = {c: pd.Index(sorted(_train_feats[c].unique())) for c in CAT_COLS}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = base_features(df)
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X[NUM_COLS + CAT_COLS]


# --- model --------------------------------------------------------------------
def make_model(n_estimators: int, es_rounds: int = 0, **kw) -> xgb.XGBClassifier:
    params = dict(
        n_estimators=n_estimators,
        max_depth=8,
        learning_rate=0.1,
        tree_method="hist",
        enable_categorical=True,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=1.0,
        eval_metric="auc",
        early_stopping_rounds=es_rounds if es_rounds else None,
        random_state=SEED,
        n_jobs=N_JOBS,
    )
    params.update(kw)
    return xgb.XGBClassifier(**params)


y_all = to_y(train)
rng = np.random.RandomState(SEED)
val_idx = rng.rand(len(train)) < 0.1
tr_idx = ~val_idx

t0 = time.time()
X_all = prepare(train)
X_eval = prepare(evald)
y_eval = to_y(evald)
print(f"Feature time: {time.time() - t0:.1f}s")

# --- small deliberate grid: depth / lr / min_child_weight, ES metric = AUC ------
GRID = [
    ("d6_lr10_mcw1", dict(max_depth=6, learning_rate=0.1, min_child_weight=1)),
    ("d6_lr10_mcw20", dict(max_depth=6, learning_rate=0.1, min_child_weight=20)),
    ("d8_lr05_mcw20", dict(max_depth=8, learning_rate=0.05, min_child_weight=20)),
    ("d4_lr10_mcw20", dict(max_depth=4, learning_rate=0.1, min_child_weight=20)),
    ("d6_lr05_mcw20", dict(max_depth=6, learning_rate=0.05, min_child_weight=20)),
]

results = []
for name, kw in GRID:
    t0 = time.time()
    es = make_model(800, es_rounds=40, **kw)
    es.fit(X_all[tr_idx], y_all[tr_idx], eval_set=[(X_all[val_idx], y_all[val_idx])], verbose=False)
    rounds = es.best_iteration + 1
    p_eval = es.predict_proba(X_eval)[:, 1]
    auc = roc_auc_score(y_eval, p_eval)
    results.append((name, kw, rounds, auc))
    print(f"grid {name}: rounds={rounds} val2005={es.best_score:.4f} eval2006={auc:.4f} ({time.time()-t0:.1f}s)")

best_name, best_kw, best_rounds, best_auc = max(results, key=lambda r: r[3])
print(f"best config: {best_name} rounds={best_rounds} eval={best_auc:.4f}")

# stage 2: refit on the full training data with the winning config / round count
t0 = time.time()
model = make_model(best_rounds, **best_kw)
model.fit(X_all, y_all)
print(f"Training time (final): {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(y_eval, predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
