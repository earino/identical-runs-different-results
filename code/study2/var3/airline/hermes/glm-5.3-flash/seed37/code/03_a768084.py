"""autoresearch XGBoost — agent-edited train.py. THIS IS THE ONLY FILE THE AGENT EDITS.

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


def _to_f32(df: pd.DataFrame, col: str) -> pd.Series:
    if df[col].dtype == object:
        return df[col].str.lstrip("c-").astype("float32")
    return pd.to_numeric(df[col], errors="coerce").astype("float32")


def add_time_feats(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    dt = pd.to_numeric(df["DepTime"], errors="coerce").astype("float64")
    out["DepTime"] = dt.astype("float32")
    hour = (dt // 100) % 24
    tod = hour + (dt % 100) / 60.0
    for name, period in (("h", 24.0), ("hh", 48.0)):
        out[f"tod_sin_{name}"] = np.sin(2 * np.pi * tod / period).astype("float32")
        out[f"tod_cos_{name}"] = np.cos(2 * np.pi * tod / period).astype("float32")
    return out


def add_cal_feats(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    m = _to_f32(df, "Month")
    dom = _to_f32(df, "DayofMonth")
    dow = _to_f32(df, "DayOfWeek")
    out["month"] = m
    out["dom"] = dom
    out["dow"] = dow
    for nm, v, period in (("m", m, 12.0), ("dom", dom, 31.0), ("w", dow, 7.0)):
        out[f"{nm}_sin"] = np.sin(2 * np.pi * v / period).astype("float32")
        out[f"{nm}_cos"] = np.cos(2 * np.pi * v / period).astype("float32")
    return out


def add_raw_feats(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["Distance"] = df["Distance"].astype("float32")
    out["log_distance"] = np.log1p(df["Distance"].astype("float64")).astype("float32")
    out["UniqueCarrier"] = df["UniqueCarrier"]
    out["Origin"] = df["Origin"]
    out["Dest"] = df["Dest"]
    return out


FEAT_BLOCKS = (add_time_feats, add_cal_feats, add_raw_feats)
CAT_COLS = ("UniqueCarrier", "Origin", "Dest")

cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this module-scope execution will NOT be applied to the hidden holdout.
    X = pd.concat([b(df) for b in FEAT_BLOCKS], axis=1)
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
def make_model(depth: int, seed: int) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=30,
        max_depth=depth,
        learning_rate=0.1,
        tree_method="hist",
        enable_categorical=True,
        random_state=seed,
        n_jobs=N_JOBS,
    )


MODELS = [(6, SEED), (3, SEED + 1), (12, SEED + 2), (18, SEED + 4)]  # (depth, seed) per ensemble member

t0 = time.time()
X_fit = prepare(train)
y_fit = to_y(train)
models = []
for depth, seed in MODELS:
    m = make_model(depth, seed)
    m.fit(X_fit, y_fit)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
