"""XGBoost binary classifier: predict dep_delayed_15min (Y/N) on airline data.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Layout: 40-model ensemble = 10 configs x 4 feature views.
Views: raw columns; +train-fitted group-mean departure-time deviations;
+day-of-year cyclic; +both. 7 of 10 configs carry a monotone constraint
(delay rises with DepTime — stable across 2005/2006).
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

M_DT = (1, 0, 0, 0, 0, 0, 0, 1)                  # monotone: delay rises with DepTime
M_DT_GRP = (1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0)  # same for the 12-col views


def prepare(df: pd.DataFrame, grp: bool = False, doy: bool = False) -> pd.DataFrame:
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
    if doy:
        mon = df.Month.str.slice(2).astype(int).to_numpy(dtype=np.float64)
        dom = df.DayofMonth.str.slice(2).astype(int).to_numpy(dtype=np.float64)
        doyv = (mon - 1) * 30.44 + dom
        X["doy_sin"] = np.sin(2 * np.pi * doyv / 365.25)
        X["doy_cos"] = np.cos(2 * np.pi * doyv / 365.25)
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
VIEWS = [(False, False), (True, False), (False, True), (True, True)]


def _new_model(overrides: dict, mono) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(**{**PARAMS, **overrides, "monotone_constraints": mono})


models = []  # (model, (grp, doy))
for i, kw in enumerate(CONFIGS):
    mono_base = M_DT if i < N_MONO else None
    for grp, doy in VIEWS:
        n_feat = 8 + (4 if grp else 0) + (2 if doy else 0)
        mono = tuple([1] + [0] * (n_feat - 1)) if i < N_MONO else tuple([0] * n_feat)
        models.append((_new_model(kw, mono), (grp, doy)))

t0 = time.time()
ytr = to_y(train)
FRAMES = {(g, d): (prepare(train, g, d), prepare(evald, g, d)) for g, d in VIEWS}
for m, view in models:
    m.fit(FRAMES[view][0], ytr)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    preds = [m.predict_proba(prepare(df, g, d))[:, 1] for m, (g, d) in models]
    return np.mean(preds, axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
