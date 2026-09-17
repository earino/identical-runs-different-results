"""XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

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
N_BAG = 12

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- features -----------------------------------------------------------------
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}

y_tr = (train[TARGET] == POSITIVE).astype(float)
PRIOR = float(y_tr.mean())


def _fit(keys: pd.DataFrame, m: float) -> pd.Series:
    g = keys.assign(y=y_tr.to_numpy()).groupby(list(keys.columns))["y"].agg(["sum", "count"])
    enc = (g["sum"] + m * PRIOR) / (g["count"] + m)
    enc.index = [k if isinstance(k, tuple) else (k,) for k in enc.index]
    return enc


_OH = train["Origin"].astype(str) + "|" + (pd.to_numeric(train["DepTime"], errors="coerce") // 100).astype(int).astype(str)
_DH = train["Dest"].astype(str) + "|" + (pd.to_numeric(train["DepTime"], errors="coerce") // 100).astype(int).astype(str)
TE_OH = _fit(pd.DataFrame({"k": _OH}), 100.0)
TE_DH = _fit(pd.DataFrame({"k": _DH}), 100.0)


def _map(enc: pd.Series, keys: pd.Series) -> np.ndarray:
    v = enc.reindex(pd.Index(keys.to_numpy())).to_numpy(dtype=float)
    return np.where(np.isnan(v), PRIOR, v)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Full view: raw features + hour + origin/dest-hour target encodings."""
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    dep = pd.to_numeric(df["DepTime"], errors="coerce")
    X["hour"] = dep // 100
    oh = df["Origin"].astype(str) + "|" + (dep // 100).astype(int).astype(str)
    dh = df["Dest"].astype(str) + "|" + (dep // 100).astype(int).astype(str)
    X["te_origin_hour"] = _map(TE_OH, oh)
    X["te_dest_hour"] = _map(TE_DH, dh)
    return X


def prepare_light(df: pd.DataFrame) -> pd.DataFrame:
    """Lean view: numeric/ordinal only (hour, distance, calendar numerics) — decorrelated from the cat-heavy view."""
    X = pd.DataFrame(index=df.index)
    dep = pd.to_numeric(df["DepTime"], errors="coerce")
    X["hour"] = dep // 100
    X["minute"] = dep % 100
    X["dep"] = dep
    for c, src in (("month_n", "Month"), ("dom_n", "DayofMonth"), ("dow_n", "DayOfWeek")):
        X[c] = pd.to_numeric(df[src].astype(str).str.replace("c-", "", regex=False), errors="coerce")
    X["log_dist"] = np.log1p(pd.to_numeric(df["Distance"], errors="coerce"))
    X["dep_sin"] = np.sin(2 * np.pi * dep / 2400.0)
    X["dep_cos"] = np.cos(2 * np.pi * dep / 2400.0)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- models: diverse bag on full view + bag on lean view ------------------------
Xtr = prepare(train)
Xtr_l = prepare_light(train)
ytr = to_y(train)

models = []
t0 = time.time()
for k in range(N_BAG):
    depth = [4, 5, 6, 7, 8, 10][k % 6]
    colsample = [0.6, 0.8, 1.0][k % 3]
    m = xgb.XGBClassifier(
        n_estimators=60,
        max_depth=depth,
        learning_rate=0.1,
        tree_method="hist",
        enable_categorical=True,
        random_state=SEED + k,
        subsample=0.7,
        colsample_bytree=colsample,
        n_jobs=N_JOBS,
    )
    m.fit(Xtr, ytr)
    models.append(m)
for k in range(6):
    m = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        tree_method="hist",
        random_state=SEED + 100 + k,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=N_JOBS,
    )
    m.fit(Xtr_l, ytr)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    Xl = prepare_light(df)
    n = len(df)
    ranks = []
    for i, m in enumerate(models):
        Xi = Xl if i >= N_BAG else X
        p = m.predict_proba(Xi)[:, 1]
        # ranks normalized to (0,1): monotone, so AUC is unchanged, but output is a valid probability
        ranks.append((pd.Series(p).rank().to_numpy() - 0.5) / n)
    return np.clip(np.mean(ranks, axis=0), 0.0, 1.0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
