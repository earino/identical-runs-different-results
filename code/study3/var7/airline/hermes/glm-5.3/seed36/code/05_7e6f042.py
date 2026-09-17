"""Airline delay classifier — OOF target encoding + cyclical time + tuned XGBoost.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, module-level `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).
     All feature engineering lives inside prepare(); encoders are fit on train only.
     Training rows use OUT-OF-FOLD target encodings (no leakage); unseen rows use full-train maps.
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


def _int_c(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace("c-", "", regex=False).astype(int)


def _base(df: pd.DataFrame) -> pd.DataFrame:
    """Raw -> tidy frame with numeric time features and raw categorical keys."""
    X = pd.DataFrame(index=df.index)
    month, dom, dow = _int_c(df["Month"]), _int_c(df["DayofMonth"]), _int_c(df["DayOfWeek"])
    dep = df["DepTime"].astype(int)
    hour, minute = dep // 100, dep % 100
    tod = hour * 60 + minute

    X["month"] = month
    X["month_sin"] = np.sin(2 * np.pi * month / 12)
    X["month_cos"] = np.cos(2 * np.pi * month / 12)
    X["dom"] = dom
    X["dow"] = dow
    X["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    X["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    X["hour"] = hour
    X["minute"] = minute
    X["tod"] = tod
    X["tod_sin"] = np.sin(2 * np.pi * tod / 1440)
    X["tod_cos"] = np.cos(2 * np.pi * tod / 1440)
    X["slot30"] = (tod // 30).astype(int)
    X["slot15"] = (tod // 15).astype(int)

    X["dist"] = df["Distance"].astype(float)
    X["log_dist"] = np.log1p(X["dist"])
    X["carrier"] = df["UniqueCarrier"].to_numpy()
    X["origin"] = df["Origin"].to_numpy()
    X["dest"] = df["Dest"].to_numpy()
    X["route"] = (df["Origin"].astype(str) + "_" + df["Dest"].astype(str)).to_numpy()
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


y_train = to_y(train)
B_train = _base(train)


def _keys_of(B: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "route": B["route"].astype(str),
        "origin": B["origin"].astype(str),
        "dest": B["dest"].astype(str),
        "carrier": B["carrier"].astype(str),
        "slot30": B["slot30"].astype(str),
        "slot15": B["slot15"].astype(str),
        "origin_slot": B["origin"].astype(str) + "_" + B["slot30"].astype(str),
        "dest_slot": B["dest"].astype(str) + "_" + B["slot30"].astype(str),
        "route_dow": B["route"].astype(str) + "_" + B["dow"].astype(str),
    }


KEY_NAMES = list(_keys_of(B_train).keys())
ALPHA = {"route": 100, "origin": 50, "dest": 50, "carrier": 100,
         "slot30": 50, "slot15": 100, "origin_slot": 20, "dest_slot": 20, "route_dow": 20}
CNT_KEYS = ("route", "origin", "dest")  # train-set key counts (support indicator)
N_SPLITS = 5

# Out-of-fold TE for TRAINING rows; full-train map for rows unseen at fit time (eval/holdout).
te_oof = {}   # name -> np.ndarray aligned with train row order
te_full = {}  # name -> dict
cnt_full = {}  # name -> dict (train-set support per key)
_keys_train = _keys_of(B_train)
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
prior = y_train.mean()
for name in KEY_NAMES:
    arr = _keys_train[name].to_numpy()
    oof = np.full(len(arr), np.nan)
    for tr_i, va_i in kf.split(arr):
        g = pd.DataFrame({"k": arr[tr_i], "y": y_train[tr_i]}).groupby("k")["y"].agg(["sum", "count"])
        te = (g["sum"] + ALPHA[name] * prior) / (g["count"] + ALPHA[name])
        oof[va_i] = pd.Series(arr[va_i]).map(te).to_numpy()
    g = pd.DataFrame({"k": arr, "y": y_train}).groupby("k")["y"].agg(["sum", "count"])
    te_full[name] = ((g["sum"] + ALPHA[name] * prior) / (g["count"] + ALPHA[name])).to_dict()
    cnt_full[name] = g["count"].to_dict()
    te_oof[name] = oof

CAT_COLS = ["carrier", "origin", "dest", "route", "slot30", "dow", "month", "hour"]
cat_levels = {c: pd.Index(sorted(B_train[c].dropna().unique())) for c in CAT_COLS}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering for ANY raw dataframe (train rows will get OOF TE patched after)."""
    X = _base(df)
    keys = _keys_of(X)
    for name in KEY_NAMES:
        X[f"te_{name}"] = keys[name].map(te_full[name]).astype(float).to_numpy()
    for name in CNT_KEYS:
        X[f"cnt_{name}"] = keys[name].map(cnt_full[name]).fillna(0).astype(float).to_numpy()
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])
    return X


def prepare_train() -> pd.DataFrame:
    """prepare(train) with TE columns replaced by their out-of-fold versions (positionally aligned)."""
    X = prepare(train)
    for name in KEY_NAMES:
        X[f"te_{name}"] = te_oof[name]
    return X


# --- model --------------------------------------------------------------------
rng = np.random.RandomState(SEED)
val_mask = rng.rand(len(train)) < 0.15
tr_rows, va_rows = np.where(~val_mask)[0], np.where(val_mask)[0]

X_all = prepare_train()
X_tr, X_va = X_all.iloc[tr_rows], X_all.iloc[va_rows]
y_tr, y_va = y_train[tr_rows], y_train[va_rows]

model = xgb.XGBClassifier(
    n_estimators=2000,
    learning_rate=0.1,
    max_depth=6,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    tree_method="hist",
    enable_categorical=True,
    early_stopping_rounds=50,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
best_it = model.best_iteration
print(f"Training time: {time.time() - t0:.1f}s, best_iter={best_it}")

# refit on ALL training rows at the ES-sized iteration count (sizing only needs the ES run).
# Ensemble of 5 diverse XGB models; averaged predictions reduce variance.
ENSEMBLE = [
    dict(max_depth=6, subsample=0.8, colsample_bytree=0.8, seed=42),
    dict(max_depth=8, subsample=0.7, colsample_bytree=0.7, seed=123),
    dict(max_depth=10, subsample=0.8, colsample_bytree=0.6, seed=7),
    dict(max_depth=6, subsample=0.9, colsample_bytree=0.9, seed=2024),
    dict(max_depth=12, subsample=0.75, colsample_bytree=0.5, seed=99),
]
members = []
for cfg in ENSEMBLE:
    m = xgb.XGBClassifier(
        n_estimators=int(best_it) if best_it is not None else 300,
        learning_rate=0.1,
        min_child_weight=5,
        reg_lambda=1.0,
        tree_method="hist",
        enable_categorical=True,
        random_state=cfg["seed"],
        n_jobs=N_JOBS,
        max_depth=cfg["max_depth"],
        subsample=cfg["subsample"],
        colsample_bytree=cfg["colsample_bytree"],
    )
    m.fit(X_all, y_train)
    members.append(m)
    print(f"fitted member depth={cfg['max_depth']} seed={cfg['seed']}")
print(f"Refit {len(members)} members on all {len(train)} rows")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in members], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
