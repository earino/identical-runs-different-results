"""XGBoost binary classifier for the airline delay task. THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Features (E4): dep_minute, hh_block, Month, DayofMonth, DayOfWeek, UniqueCarrier,
Origin, Dest, Distance, plus:
  - day-of-month bins (1-5=5, ..., 29+4=4; wraps within-month schedule cycle)
  - carrier x hh interaction: carrier ordered by its train-set 6-10h delay rate
  - route key: Origin+Dest (categorical, 5k levels, allowed via max_cat_to_onehot=0)
  - UniqueCarrier frequency encoded
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

CAT_COLS = ["hh_block", "Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest",
            "dom_bin", "route", "carrier_hh"]
PASS_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]

# ---- statistics fitted on TRAIN ONLY ------------------------------------------
tr = train.copy()
tr["hh_block"] = tr["DepTime"] // 100
tr["dom_num"] = tr["DayofMonth"].str.slice(2).astype(int)  # 'c-16' -> 16
tr["dom_bin"] = ((tr["dom_num"] - 1) % 7) + 1          # 1..7 cycle bin
tr["route"] = tr["Origin"].astype(str) + ">" + tr["Dest"].astype(str)
tr["carrier_hh"] = tr["UniqueCarrier"].astype(str) + "_" + tr["hh_block"].astype(str)

cat_levels = {c: pd.Index(sorted(tr[c].dropna().unique())) for c in CAT_COLS}
del tr

# carrier order by 6-10h delay rate on TRAIN only
_c = train.assign(hh=train["DepTime"] // 100)
_rate = (_c[(_c["hh"] >= 6) & (_c["hh"] <= 10)]
         .groupby("UniqueCarrier")[TARGET].apply(lambda s: (s == POSITIVE).mean()))
CARRIER_ORDER = _rate.sort_values().index.tolist()
CARRIER_TO_INT = {c: i for i, c in enumerate(CARRIER_ORDER)}

# carrier frequency on TRAIN only
CARRIER_FREQ = train["UniqueCarrier"].value_counts().to_dict()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    dt = df["DepTime"]
    X["dep_minute"] = (dt // 100) * 60 + (dt % 100)
    X["Distance"] = df["Distance"]
    X["hh_block"] = dt // 100
    X["dom_bin"] = ((df["DayofMonth"].str.slice(2).astype(int) - 1) % 7) + 1
    for c in PASS_COLS:
        X[c] = df[c]
    X["route"] = df["Origin"].astype(str) + ">" + df["Dest"].astype(str)
    X["carrier_hh"] = df["UniqueCarrier"].astype(str) + "_" + (dt // 100).astype(str)
    X["carrier_freq"] = df["UniqueCarrier"].map(CARRIER_FREQ).fillna(0).astype(float)
    X["carrier_rank"] = df["UniqueCarrier"].map(CARRIER_TO_INT).fillna(-1).astype(float)
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X[CAT_COLS + ["dep_minute", "Distance", "carrier_freq", "carrier_rank"]]


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
X_full, y_full = prepare(train), to_y(train)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_full, y_full, test_size=0.15, random_state=SEED, stratify=y_full
)

common = dict(
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_lambda=1.0,
    tree_method="hist",
    enable_categorical=True,
    max_cat_to_onehot=0,
    random_state=SEED,
    n_jobs=N_JOBS,
    eval_metric="auc",
)

t0 = time.time()
es_model = xgb.XGBClassifier(n_estimators=2000, early_stopping_rounds=50, **common)
es_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
best_n = int(es_model.best_iteration) + 1
print(f"ES run: best_iteration={best_n} time={time.time() - t0:.1f}s")

t0 = time.time()
model = xgb.XGBClassifier(n_estimators=best_n, **common)
model.fit(X_full, y_full, verbose=False)
print(f"Refit time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
