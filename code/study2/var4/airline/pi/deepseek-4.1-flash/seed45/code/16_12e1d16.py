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
from sklearn.model_selection import KFold

TASK = json.load(open("task.json"))
TARGET = TASK["target"]
POSITIVE = TASK["positive_label"]
ID_COLS = TASK.get("id_columns", [])
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- feature specification -----------------------------------------------------
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
# low-cardinality strings -> native XGBoost categoricals
native_cat_cols = []
# high-cardinality strings -> out-of-fold smoothed target encoding + frequency
te_cols = ["Origin", "Dest", "Route", "CoO", "CoD", "OM", "DM", "OH", "DH", "CoH", "MH", "OHh", "DHh"]
for c in obj_cols:
    if c not in te_cols:
        te_cols.append(c)
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in native_cat_cols}

SMOOTHING = 20.0
ytr = (train[TARGET] == POSITIVE).astype(int).to_numpy()
PRIOR = float(ytr.mean())


def _hour_bucket(df: pd.DataFrame) -> np.ndarray:
    t = df["DepTime"].to_numpy(dtype=np.int32)
    return ((((t // 100) % 24) * 60 + t % 100) // 30).astype(str)


def _hour_only(df: pd.DataFrame) -> np.ndarray:
    return (((df["DepTime"].to_numpy(dtype=np.int32) // 100) % 24)).astype(str)


def _keys(df: pd.DataFrame, c: str) -> np.ndarray:
    if c == "Route":
        return (df["Origin"].astype(str) + "_" + df["Dest"].astype(str)).to_numpy()
    if c == "CoO":
        return (df["UniqueCarrier"].astype(str) + "_" + df["Origin"].astype(str)).to_numpy()
    if c == "CoD":
        return (df["UniqueCarrier"].astype(str) + "_" + df["Dest"].astype(str)).to_numpy()
    if c == "OM":
        return (df["Origin"].astype(str) + "_" + df["Month"].astype(str)).to_numpy()
    if c == "DM":
        return (df["Dest"].astype(str) + "_" + df["Month"].astype(str)).to_numpy()
    if c == "OH":
        return (df["Origin"].astype(str) + "_" + _hour_bucket(df)).to_numpy()
    if c == "DH":
        return (df["Dest"].astype(str) + "_" + _hour_bucket(df)).to_numpy()
    if c == "CoH":
        return (df["UniqueCarrier"].astype(str) + "_" + _hour_bucket(df)).to_numpy()
    if c == "MH":
        return (df["Month"].astype(str) + "_" + _hour_bucket(df)).to_numpy()
    if c == "OHh":
        return (df["Origin"].astype(str) + "_" + _hour_only(df)).to_numpy()
    if c == "DHh":
        return (df["Dest"].astype(str) + "_" + _hour_only(df)).to_numpy()
    return df[c].to_numpy()


def fit_te(df: pd.DataFrame, y: np.ndarray):
    """Fit smoothed target-mean maps (plus frequencies) on the given data only."""
    enc, freq = {}, {}
    for c in te_cols:
        g = pd.DataFrame({"k": _keys(df, c), "y": y}).groupby("k")["y"].agg(["mean", "count"])
        enc[c] = ((g["mean"] * g["count"] + PRIOR * SMOOTHING) / (g["count"] + SMOOTHING)).to_dict()
        freq[c] = (g["count"] / len(y)).to_dict()
    return enc, freq


FULL_ENC, FULL_FREQ = fit_te(train, ytr)


def add_diffs(X: pd.DataFrame) -> pd.DataFrame:
    for a, b, name in [("OH_te", "Origin_te", "OH_diff"), ("DH_te", "Dest_te", "DH_diff"),
                       ("CoH_te", "UniqueCarrier_te", "CoH_diff"), ("MH_te", "Month_te", "MH_diff")]:
        if a in X.columns and b in X.columns:
            X[name] = X[a] - X[b]
    return X


def apply_te(X: pd.DataFrame, df: pd.DataFrame, enc, freq) -> pd.DataFrame:
    for c in te_cols:
        k = _keys(df, c)
        X[c + "_te"] = pd.Series(k).map(enc[c]).fillna(PRIOR).to_numpy()
        X[c + "_freq"] = pd.Series(k).map(freq[c]).fillna(0.0).to_numpy()
    return add_diffs(X)


def base_frame(df: pd.DataFrame) -> pd.DataFrame:
    X = df[[c for c in feature_cols if c not in obj_cols]].copy()
    t = X["DepTime"].to_numpy(dtype=np.int32)
    hh = (t // 100) % 24
    mm = t % 100
    X["dep_hour"] = hh
    X["dep_min"] = mm
    X["dep_time_min"] = hh * 60 + mm
    for c in native_cat_cols:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    return X


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = base_frame(df)
    X = apply_te(X, df, FULL_ENC, FULL_FREQ)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


def oof_training_frame(n_splits: int = 5) -> pd.DataFrame:
    """Training frame with out-of-fold target encodings to avoid leakage into the model fit."""
    X = base_frame(train)
    oof_te = {c: np.full(len(train), PRIOR) for c in te_cols}
    oof_fr = {c: np.zeros(len(train)) for c in te_cols}
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    for tr_idx, va_idx in kf.split(train):
        d_tr, d_va = train.iloc[tr_idx], train.iloc[va_idx]
        enc, freq = fit_te(d_tr, ytr[tr_idx])
        for c in te_cols:
            k = _keys(d_va, c)
            oof_te[c][va_idx] = pd.Series(k).map(enc[c]).fillna(PRIOR).to_numpy()
            oof_fr[c][va_idx] = pd.Series(k).map(freq[c]).fillna(0.0).to_numpy()
    for c in te_cols:
        X[c + "_te"] = oof_te[c]
        X[c + "_freq"] = oof_fr[c]
    return add_diffs(X)


# --- model --------------------------------------------------------------------
def make_model(seed: int) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=1200,
        max_depth=14,
        learning_rate=0.02,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        enable_categorical=True,
        random_state=seed,
        n_jobs=N_JOBS,
    )


SEEDS = [42, 2024]
X_train = oof_training_frame()
t0 = time.time()
models = []
for sd in SEEDS:
    m = make_model(sd)
    m.fit(X_train, ytr)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
