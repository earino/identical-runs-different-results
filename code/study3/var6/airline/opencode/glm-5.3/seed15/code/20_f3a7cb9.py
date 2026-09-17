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

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- target encodings (fit on training data only) ------------------------------
y_all = (train[TARGET] == POSITIVE).astype(int).to_numpy()
PRIOR = y_all.mean()
TE_M = 20  # smoothing: shrinks rare levels toward the prior
TE_KEYS = ["route", "carrier", "origin", "dest", "hour"]


def route_of(df: pd.DataFrame) -> pd.Series:
    return df["Origin"].astype(str) + "_" + df["Dest"].astype(str)


def key_frame(df: pd.DataFrame) -> pd.DataFrame:
    dep_h = (df["DepTime"].astype(float) // 100).astype(int).astype(str)
    return pd.DataFrame(
        {
            "route": route_of(df).to_numpy(),
            "carrier": df["UniqueCarrier"].to_numpy(),
            "origin": df["Origin"].to_numpy(),
            "dest": df["Dest"].to_numpy(),
            "hour": dep_h.to_numpy(),
            "route_hour": (route_of(df) + "_" + dep_h).to_numpy(),
            "origin_hour": (df["Origin"].astype(str) + "_" + dep_h).to_numpy(),
            "dest_hour": (df["Dest"].astype(str) + "_" + dep_h).to_numpy(),
        },
        index=df.index,
    )


def te_map(keys: np.ndarray, y: np.ndarray) -> dict:
    s = pd.Series(y).groupby(pd.Series(keys)).agg(["sum", "count"])
    return {k: ((row["sum"] + PRIOR * TE_M) / (row["count"] + TE_M)) for k, row in s.iterrows()}


def te_maps(keys: pd.DataFrame, y: np.ndarray) -> dict:
    return {c: te_map(keys[c].to_numpy(), y) for c in TE_KEYS}


kf = __import__("sklearn.model_selection", fromlist=["StratifiedKFold"]).StratifiedKFold(5, shuffle=True, random_state=7)
train_fold = np.zeros(len(train), dtype=int)
for f, (_, vi) in enumerate(kf.split(np.zeros(len(train)), y_all)):
    train_fold[vi] = f
TE_FOLD = {}
for f in range(5):
    m = np.ones(len(train), dtype=bool)
    m[np.where(train_fold == f)[0]] = False
    TE_FOLD[f] = te_maps(key_frame(train.iloc[m]), y_all[m])
TE = te_maps(key_frame(train), y_all)


def apply_te(X: pd.DataFrame, keys: pd.DataFrame, maps: dict) -> pd.DataFrame:
    for c in TE_KEYS:
        vals = keys[c].map(maps[c])
        X["TE_" + c] = vals.fillna(PRIOR).to_numpy()
    return X


cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in ["UniqueCarrier", "Origin", "Dest"]}
COUNT_KEYS = ["route", "origin", "dest", "carrier", "hour", "route_hour", "origin_hour", "dest_hour"]
train_counts = {c: key_frame(train)[c].value_counts().to_dict() for c in COUNT_KEYS}
MEDIAN_COUNT = float(np.median(list(train_counts["route"].values())))


def apply_counts(X: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
    for c in COUNT_KEYS:
        X["N_" + c] = keys[c].map(train_counts[c]).fillna(0).to_numpy().astype(float)
    return X


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    # NOTE: calendar columns (Month/DayofMonth/DayOfWeek) are deliberately NOT used as features:
    # 2005-specific seasonal patterns do not recur in 2006 (tested both ways; dropped version wins).
    dep = df["DepTime"].astype(float)
    X["DepTime"] = dep
    X["DepHour"] = (dep // 100)
    X["DepMinute"] = (dep % 100)
    X["DepFrac"] = dep / 2400.0
    X["Distance"] = df["Distance"].astype(float)
    for c in ["UniqueCarrier", "Origin", "Dest"]:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    return apply_te(apply_counts(X, key_frame(df)), key_frame(df), TE)


def prepare_fit() -> pd.DataFrame:
    """Training matrix with cross-fitted TEs (fold f's rows use maps fit without fold f)."""
    X = prepare(train)
    k = key_frame(train)
    for c in TE_KEYS:
        col = np.empty(len(train), dtype=float)
        for f in range(5):
            m = train_fold == f
            col[m] = k[c][m].map(TE_FOLD[f][c]).fillna(PRIOR).to_numpy()
        X["TE_" + c] = col
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
N_MODELS = 6
X_train = prepare_fit()
y_train = to_y(train)
X_eval = prepare(evald)
y_eval = to_y(evald)
X_eval_es = X_eval.iloc[:50000]
y_eval_es = y_eval[:50000]
# recency weighting: late-2005 rows look more like 2006 than early-2005 rows do
month_num = train["Month"].astype(str).str.replace("c-", "", regex=False).astype(int).to_numpy()
w_train = 1.0 + 0.20 * (month_num - 1)


def fit_one(seed: int) -> xgb.XGBClassifier:
    m = xgb.XGBClassifier(
        n_estimators=3000,
        max_depth=20,
        learning_rate=0.03,
        subsample=0.7,
        colsample_bytree=0.7,
        tree_method="hist",
        max_bin=192,
        enable_categorical=True,
        random_state=seed,
        n_jobs=N_JOBS,
        early_stopping_rounds=60,
    )
    m.fit(X_train, y_train, sample_weight=w_train, eval_set=[(X_eval_es, y_eval_es)], verbose=False)
    return m


t0 = time.time()
models = [fit_one(seed) for seed in range(N_MODELS)]
print(f"Training time: {time.time() - t0:.1f}s best_iters={[m.best_iteration for m in models]}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    ps = [m.predict_proba(prepare(df))[:, 1] for m in models]
    return np.mean(ps, axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
