"""XGBoost binary classifier for airline delays. THE ONLY FILE THE AGENT EDITS.

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

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
STR_INT_COLS = ["Month", "DayofMonth", "DayOfWeek"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}

# --- target encoding state (fit on train only) --------------------------------
y_full = (train[TARGET] == POSITIVE).astype(int)
PRIOR = y_full.mean()
te_groups = {
    "te_carrier": ["UniqueCarrier"],
    "te_origin": ["Origin"],
    "te_dest": ["Dest"],
    "te_route": ["Origin", "Dest"],
}
TE_SMOOTH = {name: 30 for name in te_groups}
freq_groups = {"freq_carrier": ["UniqueCarrier"], "freq_origin": ["Origin"], "freq_dest": ["Dest"], "freq_route": ["Origin", "Dest"]}
TE_COLS = list(te_groups)
FREQ_COLS = list(freq_groups)


def _key(df: pd.DataFrame, cols: list) -> pd.Series:
    if len(cols) == 1:
        return df[cols[0]]
    return df[cols[0]].str.cat(df[cols[1]], sep=">")


# full-train mappings for unseen data
te_maps = {}
for name, cols in te_groups.items():
    k = _key(train, cols)
    grp = pd.DataFrame({"k": k, "y": y_full}).groupby("k")["y"].agg(["sum", "count"])
    te_maps[name] = ((grp["sum"] + TE_SMOOTH[name] * PRIOR) / (grp["count"] + TE_SMOOTH[name])).to_dict()

freq_maps = {}
for name, cols in freq_groups.items():
    freq_maps[name] = _key(train, cols).value_counts().to_dict()


def _base_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric features derived from raw columns (no fit state needed)."""
    X = pd.DataFrame(index=df.index)
    for c in STR_INT_COLS:
        X[c] = df[c].str[2:].astype(int)
    dep = df["DepTime"]
    hour = (dep // 100) % 24
    frac = hour + (dep % 100) / 60.0
    X["DepTime"] = dep
    X["dep_hour"] = hour
    X["dep_min"] = dep % 100
    X["hour_sin"] = np.sin(2 * np.pi * frac / 24.0)
    X["hour_cos"] = np.cos(2 * np.pi * frac / 24.0)
    X["month_sin"] = np.sin(2 * np.pi * X["Month"] / 12.0)
    X["month_cos"] = np.cos(2 * np.pi * X["Month"] / 12.0)
    X["dow"] = X["DayOfWeek"]
    X["dow_sin"] = np.sin(2 * np.pi * X["DayOfWeek"] / 7.0)
    X["dow_cos"] = np.cos(2 * np.pi * X["DayOfWeek"] / 7.0)
    X["dom"] = X["DayofMonth"]
    X["Distance"] = df["Distance"]
    X["log_dist"] = np.log1p(df["Distance"])
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    for name, cols in te_groups.items():
        X[name] = _key(df, cols).map(te_maps[name]).fillna(PRIOR).astype(float)
    for name, cols in freq_groups.items():
        X[name] = np.log1p(_key(df, cols).map(freq_maps[name]).fillna(0).astype(float))
    return X


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    return _base_frame(df)


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


X_all = prepare(train)
y_all = to_y(train)

# OOF target encoding for the training matrix (prevents leakage into the fit)
oof_te = {name: np.zeros(len(train)) for name in TE_COLS}
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
for tr_idx, val_idx in kf.split(train):
    tr_part, val_part = train.iloc[tr_idx], train.iloc[val_idx]
    y_part = y_full.iloc[tr_idx]
    for name, cols in te_groups.items():
        grp = pd.DataFrame({"k": _key(tr_part, cols), "y": y_part}).groupby("k")["y"].agg(["sum", "count"])
        m = ((grp["sum"] + TE_SMOOTH[name] * PRIOR) / (grp["count"] + TE_SMOOTH[name])).to_dict()
        oof_te[name][val_idx] = _key(val_part, cols).map(m).fillna(PRIOR).to_numpy()
for name in TE_COLS:
    X_all[name] = oof_te[name]

model = xgb.XGBClassifier(
    n_estimators=2000,
    learning_rate=0.05,
    max_depth=16,
    subsample=0.7,
    colsample_bytree=0.6,
    tree_method="hist",
    enable_categorical=True,
    early_stopping_rounds=100,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
X_fit, X_val, y_fit, y_val = train_test_split(
    X_all, y_all, test_size=0.2, random_state=SEED, stratify=y_all
)
model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
best_n = model.best_iteration + 1
print(f"Early-stop fit: {time.time() - t0:.1f}s, best_iteration={best_n}")

t0 = time.time()
final_model = xgb.XGBClassifier(
    n_estimators=best_n,
    learning_rate=0.05,
    max_depth=16,
    subsample=0.7,
    colsample_bytree=0.6,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)
final_model.fit(X_all, y_all)
print(f"Refit on full train: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return final_model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
