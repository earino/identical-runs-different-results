"""XGBoost binary classifier on the airline dataset. THIS IS THE ONLY FILE THE AGENT EDITS.

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
ALPHA = 20  # target-encoding smoothing

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")
y = (train[TARGET] == POSITIVE).astype(int)

# --- target encodings (fit on TRAIN only) --------------------------------------
TE_COLS = ["UniqueCarrier", "Origin", "Dest", "route", "origin_hour", "dest_hour", "carrier_hour", "origin_dow",
           "origin_hour30", "dest_hour30", "carrier_hour30", "origin_hour60", "dest_hour60", "carrier_hour60"]
GLOBAL_MEAN = float(y.mean())


def te_key(df: pd.DataFrame, col: str) -> pd.Series:
    if col == "route":
        return df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    if col.startswith(("origin_hour", "dest_hour", "carrier_hour")):
        base = {"origin_hour": "Origin", "dest_hour": "Dest", "carrier_hour": "UniqueCarrier"}[
            col.replace("30", "").replace("60", "")
        ]
        step = 15 if col.endswith(("hour", "hour15")) else (30 if col.endswith("hour30") else 60)
        b = (pd.to_numeric(df["DepTime"], errors="coerce") // step).astype("Int64")
        return df[base].astype(str) + "_" + b.astype(str)
    if col == "origin_dow":
        return df["Origin"].astype(str) + "_" + df["DayOfWeek"].astype(str)
    if col == "origin_block":
        b = (pd.to_numeric(df["DepTime"], errors="coerce") // 15).astype("Int64")
        return df["Origin"].astype(str) + "_" + b.astype(str)
    return df[col]


def smoothed_map(keys: pd.Series, yy: pd.Series) -> dict:
    df = pd.DataFrame({"k": keys, "y": yy})
    g = df.groupby("k", observed=True)["y"].agg(["sum", "size"])
    return ((g["sum"] + ALPHA * GLOBAL_MEAN) / (g["size"] + ALPHA)).to_dict()


te_maps = {c: smoothed_map(te_key(train, c), y) for c in TE_COLS}

COUNT_COLS = ["Origin", "Dest", "route", "origin_block", "UniqueCarrier"]
count_maps = {c: te_key(train, c).value_counts().to_dict() for c in COUNT_COLS}

# kernel-smoothed time profiles: per entity, Gaussian-weighted delay rate over circular 15-min blocks
NB = 96
KSIG = 4.0
KK = ["Origin", "Dest", "UniqueCarrier", "route"]
KALPHA = {"Origin": ALPHA, "Dest": ALPHA, "UniqueCarrier": ALPHA, "route": 30}


def ksource(df: pd.DataFrame, c: str) -> pd.Series:
    return df[c] if c in df.columns else te_key(df, c)
_d = np.arange(NB)[:, None] - np.arange(NB)[None, :]
_c = np.minimum(_d, NB - np.abs(_d))
KWT = np.exp(-0.5 * (_c / KSIG) ** 2)
KWT = KWT / KWT.sum(axis=1, keepdims=True)  # rows sum to 1


def block_of(df: pd.DataFrame) -> pd.Series:
    return (pd.to_numeric(df["DepTime"], errors="coerce") // 15).mod(NB)


def profile_tables(df: pd.DataFrame, yy: pd.Series) -> dict:
    out = {}
    for c in KK:
        g = pd.DataFrame({"e": ksource(df, c), "b": block_of(df), "y": yy}).dropna()
        s = g.pivot_table(index="e", columns="b", values="y", aggfunc="sum").reindex(columns=range(NB)).fillna(0.0)
        n = g.pivot_table(index="e", columns="b", values="y", aggfunc="size").reindex(columns=range(NB)).fillna(0.0)
        vs = s.to_numpy() @ KWT.T + KALPHA[c] * GLOBAL_MEAN
        vn = n.to_numpy() @ KWT.T + KALPHA[c]
        vals = pd.DataFrame(vs / vn, index=s.index, columns=range(NB)).stack()
        out[c] = vals
    return out


kprof_full = profile_tables(train, y)


def kernel_te(df: pd.DataFrame, col: str, table: pd.Series) -> pd.Series:
    key = pd.DataFrame({"e": ksource(df, col), "b": block_of(df)}).dropna()
    idx = pd.MultiIndex.from_arrays([key["e"], key["b"]])
    v = table.reindex(idx).to_numpy()
    out = pd.Series(np.nan, index=df.index)
    out.loc[key.index] = v
    return out


FEATS = [
    "Month", "DayofMonth", "DayOfWeek", "DepTime", "hour", "minute",
    "tod_sin", "tod_cos", "Distance", "logDist",
] + [c + "_te" for c in TE_COLS] + [c + "_n" for c in COUNT_COLS] + [c + "_kte" for c in KK]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    X["Month"] = pd.to_numeric(df["Month"].astype(str).str.slice(2), errors="coerce")
    X["DayofMonth"] = pd.to_numeric(df["DayofMonth"].astype(str).str.slice(2), errors="coerce")
    X["DayOfWeek"] = pd.to_numeric(df["DayOfWeek"].astype(str).str.slice(2), errors="coerce")
    dt = pd.to_numeric(df["DepTime"], errors="coerce")
    X["DepTime"] = dt
    X["hour"] = dt // 100
    X["minute"] = dt % 100
    tod = (X["hour"] * 60 + X["minute"]) / 1440.0
    X["tod_sin"] = np.sin(2 * np.pi * tod)
    X["tod_cos"] = np.cos(2 * np.pi * tod)
    dist = pd.to_numeric(df["Distance"], errors="coerce")
    X["Distance"] = dist
    X["logDist"] = np.log1p(dist)
    for c in TE_COLS:
        X[c + "_te"] = te_key(df, c).map(te_maps[c]).fillna(GLOBAL_MEAN)
    for c in COUNT_COLS:
        X[c + "_n"] = np.log1p(te_key(df, c).map(count_maps[c]).fillna(0))
    for c in KK:
        X[c + "_kte"] = kernel_te(df, c, kprof_full[c])
    return X[FEATS]


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
def make_model(seed: int, depth: int = 6, sub: float = 0.8) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=800,
        max_depth=depth,
        learning_rate=0.03,
        subsample=sub,
        colsample_bytree=0.8,
        tree_method="hist",
        random_state=seed,
        n_jobs=N_JOBS,
    )


MEMBERS = ((42, 6, 0.8), (43, 6, 0.8), (44, 6, 0.9), (45, 8, 0.7), (46, 8, 0.8), (47, 5, 0.9), (48, 10, 0.7))

t0 = time.time()
X_train = prepare(train)
# out-of-fold target encoding for the training matrix (no self-leakage)
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
oof = pd.DataFrame(index=train.index, columns=[c + "_te" for c in TE_COLS], dtype=float)
for tr_idx, va_idx in kf.split(train):
    tr_part = train.iloc[tr_idx]
    for c in TE_COLS:
        m = smoothed_map(te_key(tr_part, c), y.iloc[tr_idx])
        oof.iloc[va_idx, oof.columns.get_loc(c + "_te")] = (
            te_key(train, c).iloc[va_idx].map(m).fillna(GLOBAL_MEAN).to_numpy()
        )
for c in TE_COLS:
    X_train[c + "_te"] = oof[c + "_te"].to_numpy()
for c in KK:
    X_train[c + "_kte"] = kernel_te(train, c, kprof_full[c])  # replaced below by OOF
oof_k = {c: np.empty(len(train)) for c in KK}
for tr_idx, va_idx in kf.split(train):
    tr_part = train.iloc[tr_idx]
    ktab = profile_tables(tr_part, y.iloc[tr_idx])
    for c in KK:
        oof_k[c][va_idx] = kernel_te(train, c, ktab[c]).to_numpy()[va_idx]
for c in KK:
    X_train[c + "_kte"] = oof_k[c]
models = []
for s, d, sub in MEMBERS:
    m = make_model(s, d, sub)
    m.fit(X_train, y.to_numpy())
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    Xp = prepare(df)
    return np.mean([m.predict_proba(Xp)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
