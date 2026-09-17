"""XGBoost classifier with time features + out-of-fold target encoding of
categorical interactions. THIS IS THE ONLY FILE THE AGENT EDITS."""
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
SEED = 42
TE_K = 50.0       # smoothing strength for target encoding
TE_FOLDS = 5

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- categorical levels (fit on train only) -----------------------------------
CAT_COLS = ["UniqueCarrier", "Origin", "Dest", "Month", "DayofMonth", "DayOfWeek"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}

Y_TRAIN = (train[TARGET] == POSITIVE).astype(int).to_numpy()
PRIOR = float(Y_TRAIN.mean())


def _dep_hour(df: pd.DataFrame) -> pd.Series:
    return (df["DepTime"].astype(np.int64) // 100).astype(str)


def _route(df: pd.DataFrame) -> pd.Series:
    return df["Origin"].astype(str) + "_" + df["Dest"].astype(str)


# key builders for target-encoded features
TE_KEY_FUNCS = {
    "te_carrier": lambda df: df["UniqueCarrier"].astype(str),
    "te_origin": lambda df: df["Origin"].astype(str),
    "te_dest": lambda df: df["Dest"].astype(str),
    "te_origin_hour": lambda df: df["Origin"].astype(str) + "_" + _dep_hour(df),
    "te_dest_hour": lambda df: df["Dest"].astype(str) + "_" + _dep_hour(df),
    "te_carrier_hour": lambda df: df["UniqueCarrier"].astype(str) + "_" + _dep_hour(df),
    "te_route_hour": lambda df: _route(df) + "_" + _dep_hour(df),
    "te_carrier_route_hour": lambda df: df["UniqueCarrier"].astype(str) + "_" + _route(df) + "_" + _dep_hour(df),
}
TE_COLS = list(TE_KEY_FUNCS)


def _smoothed_map(keys: pd.Series, y: np.ndarray, k: float) -> pd.Series:
    g = pd.DataFrame({"k": keys.values, "y": y}).groupby("k")["y"].agg(["sum", "count"])
    return (g["sum"] + PRIOR * k) / (g["count"] + k)


def _oof_encode() -> dict:
    """Out-of-fold target encoding for the training rows (avoids leakage)."""
    oof = {}
    for name, fn in TE_KEY_FUNCS.items():
        keys = fn(train).reset_index(drop=True)
        vals = np.full(len(keys), PRIOR, dtype=np.float32)
        kf = KFold(TE_FOLDS, shuffle=True, random_state=SEED)
        for a, b in kf.split(keys):
            sm = _smoothed_map(keys.iloc[a], Y_TRAIN[a], TE_K)
            vals[b] = keys.iloc[b].map(sm).fillna(PRIOR).to_numpy()
        oof[name] = vals
    return oof


# full-train mappings used at inference time (predict_proba on unseen rows)
TE_MAPS = {name: _smoothed_map(fn(train).reset_index(drop=True), Y_TRAIN, TE_K)
           for name, fn in TE_KEY_FUNCS.items()}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """ALL feature engineering lives here: raw DataFrame -> model matrix."""
    X = pd.DataFrame(index=df.index)
    d = df["DepTime"].astype(np.int64).to_numpy()
    hour = np.clip(d // 100, 0, 26)
    minute = np.clip(d % 100, 0, 59)
    mins = (hour * 60 + minute) % 1440
    X["dep_hour"] = hour
    X["dep_min"] = minute
    X["dep_mins"] = mins
    X["dep_sin"] = np.sin(2 * np.pi * mins / 1440.0)
    X["dep_cos"] = np.cos(2 * np.pi * mins / 1440.0)
    X["red_eye"] = (hour >= 24).astype(np.int8)
    X["Distance"] = df["Distance"].astype(np.float32)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    for name, fn in TE_KEY_FUNCS.items():
        keys = fn(df).reset_index(drop=True)
        X[name] = keys.map(TE_MAPS[name]).fillna(PRIOR).astype(np.float32).to_numpy()
    return X


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=450,
    max_depth=6,
    learning_rate=0.03,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
X_train = prepare(train)
for name, vals in _oof_encode().items():
    X_train[name] = vals
model.fit(X_train, Y_TRAIN)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score((evald[TARGET] == POSITIVE).astype(int), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
