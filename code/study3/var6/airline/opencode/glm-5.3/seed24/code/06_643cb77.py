"""XGBoost binary classifier for airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract:
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. Module-level `predict_proba(df)`: raw DataFrame -> 1-D array of P(positive).
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
y_train = (train[TARGET] == POSITIVE).astype(int)
PRIOR = float(y_train.mean())

CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}
KEY_COLS = [c for c in train.columns if c != TARGET]


def add_route(df: pd.DataFrame) -> pd.Series:
    return df["Origin"].str.cat(df["Dest"], sep="_")


# --- target-encoding tables, fit on train only --------------------------------
TE_SPECS = {
    "route": 30.0,
    "Origin": 100.0,
    "Dest": 100.0,
    "UniqueCarrier": 200.0,
    "hour_te": 30.0,
    "oh": 60.0,
    "ch": 120.0,
    "month_te": 300.0,
    "dow_te": 300.0,
}


def group_keys(df: pd.DataFrame) -> dict:
    keys = {}
    keys["route"] = add_route(df)
    keys["Origin"] = df["Origin"]
    keys["Dest"] = df["Dest"]
    keys["UniqueCarrier"] = df["UniqueCarrier"]
    keys["hour_te"] = (df["DepTime"].astype(int) // 100).astype(str)
    keys["oh"] = df["Origin"].str.cat(keys["hour_te"], sep="@")
    keys["ch"] = df["UniqueCarrier"].str.cat(keys["hour_te"], sep="@")
    keys["month_te"] = df["Month"]
    keys["dow_te"] = df["DayOfWeek"]
    return keys


train_keys = group_keys(train)

# OOF folds so training rows get out-of-fold encodings (no in-sample leakage).
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
train_fold = np.zeros(len(train), dtype=int)
for f, (_, val_idx) in enumerate(skf.split(train, y_train)):
    train_fold[val_idx] = f

# per-fold-group stats: stats[(spec, fold)][group] = (sum, count)
fold_stats = {}
for spec in TE_SPECS:
    for f in range(5):
        mask = train_fold != f
        g = train_keys[spec][mask]
        s = pd.Series(y_train.to_numpy()[mask]).groupby(g.values).agg(["sum", "count"])
        fold_stats[(spec, f)] = s
    full = pd.Series(y_train.to_numpy()).groupby(train_keys[spec].values).agg(["sum", "count"])
    fold_stats[(spec, -1)] = full

# train row key -> fold, so prepare() can give train rows their OOF value
train_key_to_fold = {k: v for k, v in zip(train[KEY_COLS].itertuples(index=False, name=None), train_fold)}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering lives here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    X["Month"] = df["Month"].str.slice(2).astype(int)
    X["DayofMonth"] = df["DayofMonth"].str.slice(2).astype(int)
    X["DayOfWeek"] = df["DayOfWeek"].str.slice(2).astype(int)
    dep = df["DepTime"].astype(int)
    X["DepTime"] = dep
    hour = dep // 100
    X["hour"] = hour
    X["minute"] = dep % 100
    X["Distance"] = df["Distance"].astype(float)
    X["log_distance"] = np.log1p(X["Distance"])
    month_num = X["Month"]
    dom = X["DayofMonth"]
    X["day_of_year"] = (month_num - 1) * 31 + dom
    X["doy_sin"] = np.sin(2 * np.pi * X["day_of_year"] / 366.0)
    X["doy_cos"] = np.cos(2 * np.pi * X["day_of_year"] / 366.0)
    h24 = hour % 24
    mins = h24 * 60 + X["minute"]
    X["hour_sin"] = np.sin(2 * np.pi * mins / 1440.0)
    X["hour_cos"] = np.cos(2 * np.pi * mins / 1440.0)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])

    keys = group_keys(df)
    row_fold = np.full(len(df), -1, dtype=int)
    for i, k in enumerate(df[KEY_COLS].itertuples(index=False, name=None)):
        f = train_key_to_fold.get(k)
        if f is not None:
            row_fold[i] = f
    for spec, m in TE_SPECS.items():
        full = fold_stats[(spec, -1)]
        enc_full = (full["sum"] + m * PRIOR) / (full["count"] + m)
        vals = enc_full.reindex(keys[spec].values).fillna(PRIOR).to_numpy()
        # OOF for train rows
        for f in range(5):
            idx = np.where(row_fold == f)[0]
            if len(idx):
                st = fold_stats[(spec, f)]
                enc_f = (st["sum"] + m * PRIOR) / (st["count"] + m)
                vals[idx] = enc_f.reindex(keys[spec].values[idx]).fillna(PRIOR).to_numpy()
        X[f"te_{spec}"] = vals
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


N_SEEDS = 5


def _make(seed: int) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=4000,
        max_depth=12,
        learning_rate=0.03,
        subsample=0.5,
        colsample_bytree=0.5,
        min_child_weight=5,
        reg_alpha=5,
        tree_method="hist",
        enable_categorical=True,
        early_stopping_rounds=100,
        random_state=seed,
        n_jobs=N_JOBS,
    )


X_train, y_train_all = prepare(train), to_y(train)
X_eval, y_eval_all = prepare(evald), to_y(evald)

t0 = time.time()
models = []
for s in range(N_SEEDS):
    m = _make(SEED + s)
    m.fit(X_train, y_train_all, eval_set=[(X_eval, y_eval_all)], verbose=False)
    models.append(m)
    print(f"  seed {s}: best_iter={m.best_iteration} ({time.time() - t0:.1f}s)")
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
