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


# --- model ensemble -----------------------------------------------------------
COMMON = dict(
    n_estimators=1500,
    max_depth=20,
    max_leaves=1023,
    learning_rate=0.02,
    min_child_weight=2,
    subsample=0.9,
    colsample_bytree=0.3,
    reg_lambda=10.0,
    reg_alpha=0.0,
    gamma=0.5,
    max_bin=128,
    max_cat_to_onehot=1,
    grow_policy="lossguide",
    tree_method="hist",
    enable_categorical=True,
    n_jobs=N_JOBS,
    early_stopping_rounds=50,
    eval_metric="auc",
)
SEEDS = [42, 7, 1]
models = []
BEST_ITS = []
Xtr, ytr = prepare(train), to_y(train)
Xev, yev = prepare(evald), to_y(evald)
t0 = time.time()
for s in SEEDS:
    m = xgb.XGBClassifier(random_state=s, **COMMON)
    m.fit(Xtr, ytr, eval_set=[(Xev, yev)], verbose=False)
    models.append(m)
    BEST_ITS.append(m.best_iteration + 1)
    print(f"  model seed={s} best_iter={m.best_iteration} best_auc={m.best_score:.4f}")
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    ps = [m.predict_proba(X, iteration_range=(0, bi))[:, 1] for m, bi in zip(models, BEST_ITS)]
    return np.mean(ps, axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
