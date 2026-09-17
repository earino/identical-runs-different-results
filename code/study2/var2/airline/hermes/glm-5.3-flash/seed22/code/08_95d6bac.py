"""XGBoost binary classifier: predict dep_delayed_15min (Y/N) on airline data.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Layout: a 20-model ensemble = {7 monotone-constrained + 3 unconstrained} x
{base frame, base + train-fitted group-mean departure-time deviations}.
Averaging both views beat either alone.
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

BASE_COLS = ["Month", "DayofMonth", "DayOfWeek", "DepTime", "UniqueCarrier", "Origin", "Dest", "Distance"]
CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]

# category levels and group means fitted on TRAIN only
CAT_LEVELS = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}
GRP_MAPS = {
    "dep_m_carrier": train.groupby("UniqueCarrier").DepTime.mean(),
    "dep_m_origin": train.groupby("Origin").DepTime.mean(),
    "dep_m_dest": train.groupby("Dest").DepTime.mean(),
    "dep_m_route": train.assign(_rt=train.Origin.str.cat(train.Dest, sep="_"))
                        .groupby("_rt").DepTime.mean(),
}

M_DT = (1, 0, 0, 0, 0, 0, 0, 0)                  # monotone: delay rises with DepTime
M_DT_GRP = (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)  # same for the 12-col group-deviation frame


def prepare(df: pd.DataFrame, grp: bool = False) -> pd.DataFrame:
    """ALL feature engineering lives here; predict_proba calls it on unseen rows.

    Pure transform: every statistic used (category levels, group means) was
    fitted on the training set only.
    """
    X = df[BASE_COLS].copy()
    if grp:
        X["dep_m_carrier"] = df.DepTime - df.UniqueCarrier.map(GRP_MAPS["dep_m_carrier"])
        X["dep_m_origin"] = df.DepTime - df.Origin.map(GRP_MAPS["dep_m_origin"])
        X["dep_m_dest"] = df.DepTime - df.Dest.map(GRP_MAPS["dep_m_dest"])
        X["dep_m_route"] = df.DepTime - (df.Origin.str.cat(df.Dest, sep="_")).map(GRP_MAPS["dep_m_route"])
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=CAT_LEVELS[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


PARAMS = dict(
    n_estimators=120,
    max_depth=4,
    learning_rate=0.05,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

# member configs; the first 7 also get the monotone constraint
CONFIGS = [
    dict(),
    dict(max_depth=3),
    dict(max_depth=5, n_estimators=60, learning_rate=0.1),
    dict(colsample_bynode=0.7, random_state=1),
    dict(colsample_bynode=0.7, random_state=2),
    dict(colsample_bynode=0.7, random_state=3),
    dict(colsample_bynode=0.7, random_state=4),
    dict(max_depth=3),
    dict(max_depth=5, n_estimators=60, learning_rate=0.1),
    dict(n_estimators=60, learning_rate=0.1),
]
N_MONO = 7


def _new_model(overrides: dict, mono: tuple) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(**{**PARAMS, **overrides, "monotone_constraints": mono})


models = []  # (model, use_grp_features)
for i, kw in enumerate(CONFIGS):
    mono = M_DT if i < N_MONO else None
    models.append((_new_model(kw, mono), False))
    models.append((_new_model(kw, M_DT_GRP if mono is not None else None), True))

t0 = time.time()
ytr = to_y(train)
X_base = prepare(train, grp=False)
X_grp = prepare(train, grp=True)
for m, use_grp in models:
    m.fit(X_grp if use_grp else X_base, ytr)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    Xb = prepare(df, grp=False)
    Xg = prepare(df, grp=True)
    preds = [m.predict_proba(Xg if use_grp else Xb)[:, 1] for m, use_grp in models]
    return np.mean(preds, axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
