"""XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS."""
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

# --- feature definitions ------------------------------------------------------
CATS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
# categorical levels from TRAIN ONLY; unseen levels -> NaN inside prepare()
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CATS}
cat_levels["TODbin"] = pd.Index(range(48))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering lives here so predict_proba reproduces it on unseen rows.
    X = pd.DataFrame(index=df.index)
    for c in CATS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    h = (df["DepTime"].astype("int64") // 100) % 24
    m = (df["DepTime"].astype("int64") % 100).clip(upper=59)
    tod = h * 60 + m
    X["TOD"] = tod.astype("int32")
    X["H"] = h.astype("int16")
    X["TODbin"] = pd.Categorical(tod // 30, categories=cat_levels["TODbin"])
    X["TOD_sin"] = np.sin(2 * np.pi * tod / 1440.0)
    X["TOD_cos"] = np.cos(2 * np.pi * tod / 1440.0)
    X["Dist"] = df["Distance"].astype("float32")
    X["LogD"] = np.log1p(df["Distance"].astype("float32"))
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=2500,
    max_depth=20,
    max_leaves=511,
    learning_rate=0.03,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.3,
    reg_lambda=10.0,
    reg_alpha=0.0,
    gamma=1.0,
    max_bin=96,
    max_cat_to_onehot=1,
    grow_policy="lossguide",
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
    early_stopping_rounds=60,
    eval_metric="auc",
)

t0 = time.time()
model.fit(prepare(train), to_y(train), eval_set=[(prepare(evald), to_y(evald))], verbose=False)
print(f"Training time: {time.time() - t0:.1f}s  best_iter={model.best_iteration}")
BEST_IT = model.best_iteration + 1


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df), iteration_range=(0, BEST_IT))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
