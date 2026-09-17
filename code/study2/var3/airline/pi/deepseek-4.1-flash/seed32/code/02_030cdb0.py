"""XGBoost binary classifier for airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

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
from sklearn.model_selection import KFold, train_test_split

TASK = json.load(open("task.json"))
TARGET = TASK["target"]
POSITIVE = TASK["positive_label"]
ID_COLS = TASK.get("id_columns", [])
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42
TE_ALPHA = 100.0

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- features -----------------------------------------------------------------
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}


def _hour(df: pd.DataFrame) -> pd.Series:
    return ((df["DepTime"].to_numpy() // 100) % 24).astype(int).astype(str)


def _te_keys(df: pd.DataFrame, name: str) -> pd.Series:
    h = _hour(df)
    if name == "Route":
        return df["Origin"].astype(str) + "|" + df["Dest"].astype(str)
    if name == "CarrierHour":
        return df["UniqueCarrier"].astype(str) + "|" + h
    if name == "OriginHour":
        return df["Origin"].astype(str) + "|" + h
    if name == "RouteHour":
        return df["Origin"].astype(str) + "|" + df["Dest"].astype(str) + "|" + h
    if name == "Carrier":
        return df["UniqueCarrier"].astype(str)
    return df[name].astype(str)


TE_NAMES = ["Route", "Carrier", "Origin", "Dest", "CarrierHour", "OriginHour"]
PRIOR = (train[TARGET] == POSITIVE).mean()


def _te_fit(keys: pd.Series, y: np.ndarray) -> pd.Series:
    g = pd.DataFrame({"k": keys.to_numpy(), "y": y}).groupby("k")["y"].agg(["sum", "count"])
    return (g["sum"] + PRIOR * TE_ALPHA) / (g["count"] + TE_ALPHA)


# Full-train maps used at inference time (never fit on the incoming dataframe).
TE_MAPS = {name: _te_fit(_te_keys(train, name), (train[TARGET] == POSITIVE).astype(int).to_numpy()) for name in TE_NAMES}


def _add_te(X: pd.DataFrame, df: pd.DataFrame, oof_y: np.ndarray = None) -> pd.DataFrame:
    """If oof_y is given, produce out-of-fold target encodings (training); else use full-train maps (inference)."""
    for name in TE_NAMES:
        keys = _te_keys(df, name)
        if oof_y is None:
            X["te_" + name] = keys.map(TE_MAPS[name]).fillna(PRIOR).to_numpy()
        else:
            oof = np.full(len(df), PRIOR)
            kf = KFold(5, shuffle=True, random_state=SEED)
            for a, b in kf.split(oof):
                m = _te_fit(keys.iloc[a], oof_y[a])
                oof[b] = keys.iloc[b].map(m).fillna(PRIOR).to_numpy()
            X["te_" + name] = oof
    return X


def _base(df: pd.DataFrame) -> pd.DataFrame:
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])
    t = X["DepTime"].to_numpy()
    X["dep_hour"] = (t // 100) % 24
    X["dep_minute"] = t % 100
    return X


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # Inference path used by predict_proba on unseen rows.
    return _add_te(_base(df), df, None)


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


y = to_y(train)
X = _add_te(_base(train), train, y)

# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=3000,
    max_depth=5,
    learning_rate=0.02,
    subsample=0.6,
    colsample_bytree=0.5,
    min_child_weight=100,
    reg_lambda=20.0,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
    early_stopping_rounds=100,
)

t0 = time.time()
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.15, random_state=SEED, stratify=y)
model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
print(f"Training time: {time.time() - t0:.1f}s  best_iter={model.best_iteration}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
