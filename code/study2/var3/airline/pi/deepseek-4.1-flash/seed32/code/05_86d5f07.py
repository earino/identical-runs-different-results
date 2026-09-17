"""XGBoost binary classifier for airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Feature engineering: smoothed target encoding of route / carrier / airport / carrier-hour / origin-hour,
computed out-of-fold on the training data and with full-training-set maps at inference time.  The model is
heavily regularized because train (2005) and eval (2006) are a year apart: an unregularized model memorizes
2005 and generalizes worse.
"""
import json
import os
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

TASK = json.load(open("task.json"))
TARGET = TASK["target"]
POSITIVE = TASK["positive_label"]
ID_COLS = TASK.get("id_columns", [])
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 0
TE_ALPHA = 200.0
NFOLD = 5

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

y = (train[TARGET] == POSITIVE).astype(int).to_numpy()
PRIOR = y.mean()

# --- columns ------------------------------------------------------------------
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}

TE_SPECS = {
    "Route": ["Origin", "Dest"],
    "Carrier": ["UniqueCarrier"],
    "Origin": ["Origin"],
    "Dest": ["Dest"],
    "CarrierHour": ["UniqueCarrier", "__hour"],
    "OriginHour": ["Origin", "__hour"],
    "RouteHour": ["Origin", "Dest", "__hour"],
    "CarrierOrigin": ["UniqueCarrier", "Origin"],
    "CarrierRouteHour": ["UniqueCarrier", "Origin", "Dest", "__hour"],
}
# Frequency (count) encodings are computed for these keys from the training set.
COUNT_NAMES = [n for n in TE_SPECS if n != "CarrierRouteHour"]


def _te_keys(df: pd.DataFrame, name: str) -> pd.Series:
    cols = TE_SPECS[name]
    h = ((df["DepTime"].to_numpy() // 100) % 24).astype(str)
    parts = [h if c == "__hour" else df[c].astype(str) for c in cols]
    out = parts[0]
    for p in parts[1:]:
        out = out + "|" + p
    return out


def _te_fit(keys: pd.Series, yv: np.ndarray) -> pd.Series:
    g = pd.DataFrame({"k": keys, "y": yv}).groupby("k")["y"].agg(["sum", "count"])
    return (g["sum"] + PRIOR * TE_ALPHA) / (g["count"] + TE_ALPHA)


# Full-training-set maps; used at inference time only (never fit on the incoming dataframe).
TE_MAPS = {name: _te_fit(_te_keys(train, name), y) for name in TE_SPECS}
COUNT_MAPS = {name: _te_keys(train, name).value_counts() for name in COUNT_NAMES}


def _te_oof(keys: pd.Series) -> np.ndarray:
    """Out-of-fold target encoding so the model does not see a row's own label."""
    oof = np.full(len(keys), PRIOR)
    kf = KFold(NFOLD, shuffle=True, random_state=SEED)
    for a, b in kf.split(keys):
        m = _te_fit(keys.iloc[a], y[a])
        oof[b] = keys.iloc[b].map(m).fillna(PRIOR).to_numpy()
    return oof


def _base(df: pd.DataFrame) -> pd.DataFrame:
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    t = X["DepTime"].to_numpy()
    X["dep_hour"] = (t // 100) % 24
    X["dep_minute"] = t % 100
    return X


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # Inference path used by predict_proba on unseen rows.
    X = _base(df)
    for name in TE_SPECS:
        X["te_" + name] = _te_keys(df, name).map(TE_MAPS[name]).fillna(PRIOR).to_numpy()
    for name in COUNT_NAMES:
        X["cnt_" + name] = _te_keys(df, name).map(COUNT_MAPS[name]).fillna(0).to_numpy()
    return X


X = _base(train)
for name in TE_SPECS:
    X["te_" + name] = _te_oof(_te_keys(train, name))
for name in COUNT_NAMES:
    X["cnt_" + name] = _te_keys(train, name).map(COUNT_MAPS[name]).fillna(0).to_numpy()

# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=600,
    max_depth=7,
    learning_rate=0.02,
    subsample=0.6,
    colsample_bytree=0.5,
    min_child_weight=50,
    reg_lambda=10.0,
    tree_method="hist",
    enable_categorical=True,
    random_state=42,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(X, y)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score((evald[TARGET] == POSITIVE).astype(int).to_numpy(), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
