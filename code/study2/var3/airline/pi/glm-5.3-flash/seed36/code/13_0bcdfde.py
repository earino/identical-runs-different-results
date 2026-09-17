"""Baseline XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).
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

# --- features -----------------------------------------------------------------
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
# baseline: treat string columns as categoricals, but drop very high-cardinality ones (names, free text, timestamps)
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}
y_train = (train[TARGET] == POSITIVE).astype(int)
PRIOR = y_train.mean()
SMOOTH = 20.0
te_maps = {}
for c in ["Origin", "Dest", "UniqueCarrier"]:
    g = train.groupby(c, observed=True)[TARGET].apply(lambda s: (s == POSITIVE).mean())
    cnt = train[c].value_counts()
    te_maps[c] = ((g * cnt + SMOOTH * PRIOR) / (cnt + SMOOTH)).to_dict()


def _tod(t: pd.Series) -> pd.Series:
    """hhmm int -> minutes of day (2400+ wraps to early morning)."""
    t = t.astype("int64")
    return ((t // 100) % 24) * 60 + (t % 100)


_hr = _tod(train["DepTime"]) // 60
TE_HOUR = ((train.groupby(_hr)[TARGET].apply(lambda s: (s == POSITIVE).mean()) * _hr.value_counts()
            + SMOOTH * PRIOR) / (_hr.value_counts() + SMOOTH))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    minutes = _tod(df["DepTime"])
    X["deptime_sin"] = np.sin(2 * np.pi * minutes / 1440.0)
    X["deptime_cos"] = np.cos(2 * np.pi * minutes / 1440.0)
    X["minutes"] = minutes
    for c in ["Month", "DayofMonth", "DayOfWeek"]:
        X[c + "_num"] = pd.to_numeric(df[c].str.replace("c-", "", regex=False), errors="coerce")
    X["log_distance"] = np.log1p(df["Distance"])
    md = (X["Month_num"] * 100 + X["DayofMonth_num"])
    peak = ((md.between(1215, 1231)) | md.isin([701, 702, 703, 704, 1123, 1124, 1125, 1126, 1127, 1128,
                                                218, 219, 220, 221, 222, 527, 528, 529, 530, 531, 630]))
    X["holiday_peak"] = peak.astype(int)
    for c, m in te_maps.items():
        X["te_" + c] = df[c].map(m).fillna(PRIOR).astype(float)
    X["te_hour"] = ((_tod(df["DepTime"]) // 60).map(TE_HOUR)).fillna(PRIOR).astype(float)
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
base = dict(
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

X_all = prepare(train)
y_all = to_y(train)
t0 = time.time()
configs = [
    dict(n_estimators=400, max_depth=3, learning_rate=0.05),
    dict(n_estimators=600, max_depth=4, learning_rate=0.03),
    dict(n_estimators=800, max_depth=5, learning_rate=0.03),
    dict(n_estimators=600, max_depth=5, learning_rate=0.03),
    dict(n_estimators=800, max_depth=2, learning_rate=0.03),
]
models = [xgb.XGBClassifier(**{**base, **cfg}) for cfg in configs]
for m in models:
    m.fit(X_all, y_all)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
