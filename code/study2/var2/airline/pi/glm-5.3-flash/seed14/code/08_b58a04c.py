"""Airline delay XGBoost. Exp 10: 3-seed ensemble."""
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

# --- fitted statistics (train only) -------------------------------------------
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}
CH_TRAIN = train["UniqueCarrier"] + "_" + ((train["DepTime"].astype(float) // 100).mod(24)).astype(int).astype(str)
CH_LEVELS = pd.Index(sorted(CH_TRAIN.unique()))
ORIG_CNT = train["Origin"].value_counts().to_dict()
DEST_CNT = train["Dest"].value_counts().to_dict()
ORH_CNT = (train["Origin"] + "_" + CH_TRAIN.str.split("_", n=1).str[1]).value_counts().to_dict()


def prep_base(df: pd.DataFrame) -> pd.DataFrame:
    """All feature engineering lives here (called by predict_proba too)."""
    X = pd.DataFrame(index=df.index)
    X["month"] = df["Month"].str.replace("c-", "", regex=False).astype(int)
    X["dom"] = df["DayofMonth"].str.replace("c-", "", regex=False).astype(int)
    X["dow"] = df["DayOfWeek"].str.replace("c-", "", regex=False).astype(int)
    deptime = df["DepTime"].astype(float)
    hour = (deptime // 100).mod(24)
    minute = deptime.mod(100)
    tod = hour * 60 + minute
    ang = 2 * np.pi * tod / 1440.0
    X["deptime"] = deptime
    X["dep_hour"] = hour
    X["dep_minute"] = minute
    X["tod_sin"] = np.sin(ang)
    X["tod_cos"] = np.cos(ang)
    X["distance"] = df["Distance"].astype(float)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    ch = df["UniqueCarrier"] + "_" + ((df["DepTime"].astype(float) // 100).mod(24)).astype(int).astype(str)
    X["carrier_hour"] = pd.Categorical(ch, categories=CH_LEVELS)
    X["orig_cnt"] = df["Origin"].map(ORIG_CNT).astype(float)
    X["dest_cnt"] = df["Dest"].map(DEST_CNT).astype(float)
    X["orh_cnt"] = (df["Origin"] + "_" + ((df["DepTime"].astype(float) // 100).mod(24)).astype(int).astype(str)).map(ORH_CNT).astype(float)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
def make_model(seed: int) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=2500,
        learning_rate=0.02,
        max_depth=12,
        min_child_weight=5,
        subsample=0.9,
        colsample_bytree=0.8,
        tree_method="hist",
        enable_categorical=True,
        eval_metric="auc",
        early_stopping_rounds=150,
        random_state=seed,
        n_jobs=N_JOBS,
    )

SEEDS = [42, 7, 2025]
t0 = time.time()
eval_set = [(prep_base(evald), to_y(evald))]
models = []
for seed in SEEDS:
    m = make_model(seed)
    m.fit(prep_base(train), to_y(train), eval_set=eval_set, verbose=False)
    models.append(m)
    print(f"seed {seed}: best_iter={m.best_iteration} auc={m.best_score:.4f}")
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    P = np.mean([m.predict_proba(prep_base(df))[:, 1] for m in models], axis=0)
    return P


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
