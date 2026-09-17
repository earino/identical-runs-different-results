"""XGBoost airline-delay classifier: native-cat mini-bag + 11-member deep one-hot bag, logit-averaged.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. Module-level `predict_proba(df)`: raw DataFrame -> 1-D array of P(positive).
All feature engineering lives in prepare(); encodings/categories are fitted on data/train.csv only.
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

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}
HOUR_CATS = list(range(0, 27))
BAG_DEEP = [(0.85, 0.5, s) for s in (1, 2, 3, 4, 5, 6, 7)] + [(0.90, 0.5, 21), (0.85, 0.35, 61)]
BAG_RECENCY = [51, 52]
RECENCY_ALPHA = 0.5
BAG_NATIVE = [11, 12, 13, 14, 15]
CHUNK = 100_000


def _num(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "MonthN": df["Month"].str.replace("c-", "", regex=False).astype(int),
        "DayN": df["DayofMonth"].str.replace("c-", "", regex=False).astype(int),
        "DoWN": df["DayOfWeek"].str.replace("c-", "", regex=False).astype(int),
        "DepTime": df["DepTime"],
        "Distance": df["Distance"],
    })


def _hour_dummies(df: pd.DataFrame) -> pd.DataFrame:
    H = pd.get_dummies(pd.Categorical(df["DepTime"] // 100, categories=HOUR_CATS), dummy_na=True)
    H.columns = [f"h{c}" for c in H.columns]
    return H.astype(np.float32)


def _cat_dummies(df: pd.DataFrame) -> pd.DataFrame:
    D = pd.get_dummies(
        pd.DataFrame({c: pd.Categorical(df[c], categories=cat_levels[c]) for c in CAT_COLS}),
        dummy_na=True)
    return D.astype(np.float32)


def prepare(df: pd.DataFrame) -> dict:
    """Feature engineering for both model views. Fitted mappings come from `train` only."""
    num = _num(df)
    hd = _hour_dummies(df)
    native = num.copy()
    for c in CAT_COLS:
        native[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    native = pd.concat([native, hd], axis=1)
    onehot = pd.concat([num, _cat_dummies(df), hd], axis=1)
    return {"native": native, "onehot": onehot}


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


# --- models --------------------------------------------------------------------
y = to_y(train)
F = prepare(train)
dtrain_onehot = xgb.DMatrix(F["onehot"], label=y)

t0 = time.time()
models_native = []
for s in BAG_NATIVE:
    m = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05, tree_method="hist",
        enable_categorical=True, subsample=0.8, colsample_bytree=0.8, random_state=s, n_jobs=N_JOBS,
    )
    m.fit(F["native"], y)
    models_native.append(m)
print(f"Native bag: {time.time() - t0:.1f}s")

models_deep = []
_months = train["Month"].str.replace("c-", "", regex=False).astype(int).to_numpy()
_mnorm = (_months - _months.min()) / (_months.max() - _months.min())
_w_recency = 1 + RECENCY_ALPHA * _mnorm
for sub, col, s in BAG_DEEP:
    t0 = time.time()
    booster = xgb.train(
        {"objective": "binary:logistic", "max_depth": 20, "learning_rate": 0.075,
         "tree_method": "hist", "subsample": sub, "colsample_bytree": col,
         "random_state": s, "nthread": N_JOBS},
        dtrain_onehot, num_boost_round=200,
    )
    models_deep.append(booster)
    print(f"Deep member (sub={sub}, col={col}, seed={s}): {time.time() - t0:.1f}s")

dtrain_recency = xgb.DMatrix(F["onehot"], label=y, weight=_w_recency)
for s in BAG_RECENCY:
    t0 = time.time()
    booster = xgb.train(
        {"objective": "binary:logistic", "max_depth": 20, "learning_rate": 0.075,
         "tree_method": "hist", "subsample": 0.85, "colsample_bytree": 0.5,
         "random_state": s, "nthread": N_JOBS},
        dtrain_recency, num_boost_round=200,
    )
    models_deep.append(booster)
    print(f"Recency member (alpha={RECENCY_ALPHA}, seed={s}): {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    out = np.empty(len(df))
    n_models = len(models_native) + len(models_deep)
    for i in range(0, len(df), CHUNK):
        chunk = df.iloc[i:i + CHUNK]
        Fx = prepare(chunk)
        s = np.zeros(len(chunk))
        for m in models_native:
            s += _logit(m.predict_proba(Fx["native"])[:, 1])
        dm = xgb.DMatrix(Fx["onehot"])
        for b in models_deep:
            s += _logit(b.predict(dm))
        out[i:i + CHUNK] = 1 / (1 + np.exp(-s / n_models))
    return out


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
