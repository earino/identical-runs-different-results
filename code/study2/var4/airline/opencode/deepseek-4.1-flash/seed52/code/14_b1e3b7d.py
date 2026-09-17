"""Baseline XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

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

# --- features -----------------------------------------------------------------
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
# baseline: treat string columns as categoricals, but drop very high-cardinality ones (names, free text, timestamps)
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}
_route_tr = train["Origin"].astype(str) + "_" + train["Dest"].astype(str)
FREQ = {
    "freq_origin": (train["Origin"].astype(str).value_counts() / len(train)).to_dict(),
    "freq_dest": (train["Dest"].astype(str).value_counts() / len(train)).to_dict(),
    "freq_route": (_route_tr.value_counts() / len(train)).to_dict(),
    "freq_carrier": (train["UniqueCarrier"].astype(str).value_counts() / len(train)).to_dict(),
}
_org_tr = train["Origin"].astype(str)
_dst_tr = train["Dest"].astype(str)
_car_tr = train["UniqueCarrier"].astype(str)
FREQ["origin_n"] = _org_tr.value_counts().to_dict()
FREQ["dest_n"] = _dst_tr.value_counts().to_dict()
FREQ["route_ncarriers"] = train.groupby(_route_tr)["UniqueCarrier"].nunique().to_dict()
FREQ["carrier_origin_n"] = train.groupby([_car_tr, _org_tr]).size().to_dict()
FREQ["carrier_dest_n"] = train.groupby([_car_tr, _dst_tr]).size().to_dict()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    dt = pd.to_numeric(df["DepTime"], errors="coerce").fillna(0)
    tod = (dt // 100) % 24 + (dt % 100) / 60.0
    X["hour"] = (dt // 100) % 24
    X["tod_sin"] = np.sin(2 * np.pi * tod / 24.0)
    X["tod_cos"] = np.cos(2 * np.pi * tod / 24.0)
    month = pd.to_numeric(df["Month"].astype(str).str.replace("c-", "", regex=False), errors="coerce")
    dom = pd.to_numeric(df["DayofMonth"].astype(str).str.replace("c-", "", regex=False), errors="coerce")
    doy = pd.to_datetime(pd.DataFrame({"year": 2005, "month": month, "day": dom}), errors="coerce").dt.dayofyear
    X["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    X["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    _hol = np.array([1, 122, 187, 249, 330, 359, 365], dtype=float)
    _d = np.abs(doy.to_numpy(dtype=float)[:, None] - _hol[None, :])
    _d = np.minimum(_d, 365.25 - _d)
    X["holiday_dist"] = np.nanmin(_d, axis=1)
    route = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    X["freq_origin"] = df["Origin"].astype(str).map(FREQ["freq_origin"]).fillna(0.0)
    X["freq_dest"] = df["Dest"].astype(str).map(FREQ["freq_dest"]).fillna(0.0)
    X["freq_route"] = route.map(FREQ["freq_route"]).fillna(0.0)
    X["freq_carrier"] = df["UniqueCarrier"].astype(str).map(FREQ["freq_carrier"]).fillna(0.0)
    _car = df["UniqueCarrier"].astype(str)
    _org = df["Origin"].astype(str)
    _dst = df["Dest"].astype(str)
    _on = _org.map(FREQ["origin_n"]).fillna(0.0)
    _dn = _dst.map(FREQ["dest_n"]).fillna(0.0)
    X["route_ncarriers"] = route.map(FREQ["route_ncarriers"]).fillna(1.0)
    X["carrier_origin_share"] = pd.Series(list(zip(_car, _org)), index=df.index).map(FREQ["carrier_origin_n"]).fillna(0.0) / (_on + 1.0)
    X["carrier_dest_share"] = pd.Series(list(zip(_car, _dst)), index=df.index).map(FREQ["carrier_dest_n"]).fillna(0.0) / (_dn + 1.0)
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model: seed-averaged XGBoost ensemble ------------------------------------
Xtr = prepare(train)
ytr = to_y(train)

MODELS = []
t0 = time.time()
_cfg = []
for seed, depth in [(42, 4), (7, 4), (13, 3), (2024, 5), (123, 4), (999, 5)]:
    _cfg.append((seed, depth, 0.8, 0.8, 1.0))
for seed, depth, ss, cs in [(11, 4, 0.6, 1.0), (17, 4, 1.0, 0.6), (23, 5, 0.6, 0.6), (31, 3, 1.0, 0.8), (37, 4, 0.7, 1.0), (41, 5, 1.0, 1.0)]:
    _cfg.append((seed, depth, ss, cs, 5.0))
for seed, depth, ss, cs, mcw in _cfg:
    m = xgb.XGBClassifier(
        n_estimators=700,
        max_depth=depth,
        learning_rate=0.02,
        subsample=ss,
        colsample_bytree=cs,
        min_child_weight=mcw,
        reg_lambda=20.0,
        reg_alpha=5.0,
        gamma=3.0,
        tree_method="hist",
        enable_categorical=True,
        random_state=seed,
        n_jobs=N_JOBS,
    )
    m.fit(Xtr, ytr)
    MODELS.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in MODELS], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
