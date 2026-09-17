"""XGBoost binary classifier for airline delay prediction.

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
from sklearn.model_selection import StratifiedKFold

TASK = json.load(open("task.json"))
TARGET = TASK["target"]
POSITIVE = TASK["positive_label"]
ID_COLS = TASK.get("id_columns", [])
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- global vocabularies / statistics: fit on TRAINING data only -----------------
RARE_THRESHOLD = 300

def _rare_set(series: pd.Series, threshold: int) -> set:
    counts = series.astype(str).value_counts()
    return set(counts[counts < threshold].index)

def _rare_transform(series: pd.Series, rare_set: set) -> pd.Series:
    s = series.astype(str)
    return s.where(~s.isin(rare_set), "__RARE__")

ORIGIN_RARE = _rare_set(train["Origin"], RARE_THRESHOLD)
DEST_RARE = _rare_set(train["Dest"], RARE_THRESHOLD)
ORIGIN_COUNT = train["Origin"].astype(str).value_counts().to_dict()
DEST_COUNT = train["Dest"].astype(str).value_counts().to_dict()
_carrier_b = train["UniqueCarrier"].astype(str)
_origin_b = _rare_transform(train["Origin"], ORIGIN_RARE)
_dest_b = _rare_transform(train["Dest"], DEST_RARE)
CAT_SPECS = {"carrier": _carrier_b, "origin": _origin_b, "dest": _dest_b}
CAT_LEVELS = {k: pd.Index(sorted(v.dropna().unique())) for k, v in CAT_SPECS.items()}


def _cat_map(series: pd.Series, rare_set: set, levels: pd.Index) -> pd.Series:
    """Bucket train-rare AND never-seen categories into a shared '__RARE__' level."""
    s = series.astype(str)
    s = s.where(s.isin(levels), "__RARE__")
    return s.where(~s.isin(rare_set), "__RARE__")


def _to_int(series: pd.Series) -> pd.Series:
    """c-<n> strings -> ints."""
    return series.astype(str).str.replace("c-", "", regex=False).astype(int)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """ALL feature engineering lives here so predict_proba() can reproduce it on unseen rows."""
    mon = _to_int(df["Month"])
    dom = _to_int(df["DayofMonth"])
    dow = _to_int(df["DayOfWeek"])

    dep = df["DepTime"].astype(int)
    tod = (dep // 100) * 60 + (dep % 100)  # minutes after midnight (may exceed 1440)
    tod = np.where(tod >= 1440, tod - 1440, tod)

    X = pd.DataFrame(index=df.index)
    # dom ablated
    X["dow"] = dow.to_numpy()
    X["hour"] = (tod // 60).astype(np.float32)
    X["minute"] = (tod % 60).astype(np.float32)
    X["tod"] = tod.astype(np.float32)
    X["tod_sin"] = np.sin(2 * np.pi * tod / 1440.0).astype(np.float32)
    X["tod_cos"] = np.cos(2 * np.pi * tod / 1440.0).astype(np.float32)
    # month features ablated (weak cross-year stability)
    X["dow_sin"] = np.sin(2 * np.pi * dow / 7.0).astype(np.float32)
    X["dow_cos"] = np.cos(2 * np.pi * dow / 7.0).astype(np.float32)
    X["is_weekend"] = (dow >= 6).astype(np.int8)
    X["distance"] = df["Distance"].astype(np.float32)
    X["log_distance"] = np.log1p(df["Distance"].astype(np.float32))

    X["carrier"] = pd.Categorical(df["UniqueCarrier"].astype(str), categories=CAT_LEVELS["carrier"])
    X["origin"] = pd.Categorical(
        _cat_map(df["Origin"], ORIGIN_RARE, CAT_LEVELS["origin"]), categories=CAT_LEVELS["origin"]
    )
    X["dest"] = pd.Categorical(
        _cat_map(df["Dest"], DEST_RARE, CAT_LEVELS["dest"]), categories=CAT_LEVELS["dest"]
    )
    X["origin_count"] = np.log1p(df["Origin"].astype(str).map(ORIGIN_COUNT).fillna(0)).astype(np.float32)
    X["dest_count"] = np.log1p(df["Dest"].astype(str).map(DEST_COUNT).fillna(0)).astype(np.float32)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model: K-fold bagged XGBoost ensemble -------------------------------------
X_train = prepare(train)
y_train = to_y(train)

NFOLD = 7
skf = StratifiedKFold(n_splits=NFOLD, shuffle=True, random_state=SEED)
models = []
t0 = time.time()
for k, (fit_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    m = xgb.XGBClassifier(
        n_estimators=3000,
        max_depth=24,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.6,
        min_child_weight=5,
        reg_lambda=1.0,
        tree_method="hist",
        enable_categorical=True,
        max_cat_to_onehot=1,
        random_state=SEED + k,
        n_jobs=N_JOBS,
        early_stopping_rounds=50,
    )
    m.fit(
        X_train.iloc[fit_idx],
        y_train[fit_idx],
        eval_set=[(X_train.iloc[val_idx], y_train[val_idx])],
        verbose=False,
    )
    models.append(m)
    print(f"fold {k}: best_iter={m.best_iteration} auc={roc_auc_score(y_train[val_idx], m.predict_proba(X_train.iloc[val_idx])[:, 1]):.4f}")
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
