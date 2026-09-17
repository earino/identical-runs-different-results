"""XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Approach: train=2005, eval/holdout=2006 (time shift). What transfers across the year boundary:
  * raw DepTime (time-of-day is the dominant, structural signal; also handles dirty hhmm values
    like DepTime<100 = 00:xx and DepTime>2400 via circular minutes),
  * Origin/Dest/UniqueCarrier as categoricals (coarse effects),
  * kernel-smoothed *delay-rate profiles* over the time-of-day, one curve per Origin / Dest / Carrier,
    estimated from 2005 only, circular Gaussian kernel (sigma 8-60 min), shrunk toward the prior.
    These capture "delay-prone schedule slots" (structural airport/carrier congestion patterns).
Month/DayofMonth are dropped: 2005-specific seasonality does not transfer to 2006.
Training rows use out-of-fold profile values; eval/holdout rows use full-train profiles.
Final model: 5-seed ensemble of shallow-ish XGBoost (depth 6, 600 rounds, lr 0.03, heavy col subsampling).
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

# --- feature engineering -------------------------------------------------------
RAW_NUM = ["DepTime", "Distance"]
CATS = ["DayOfWeek", "UniqueCarrier", "Origin", "Dest"]  # Month/DayofMonth dropped (year-specific noise)
CELL = 2          # minutes per grid cell of the daily time profile
G = 1440 // CELL  # grid cells per day
N_FOLDS = 5
N_SEEDS = 3

y_train = (train[TARGET] == POSITIVE).astype(int).to_numpy()
prior = float(y_train.mean())

def minutes_arr(dep: np.ndarray) -> np.ndarray:
    """DepTime (hhmm int, with dirty values) -> minutes since midnight, circular."""
    return ((dep // 100) * 60 + (dep % 100)) % 1440

# kernel-smoothed profile specs: (name, key column, sigma minutes, shrinkage m, kind)
# kind 'dep': profile indexed by the flight's own departure time of day.
# kind 'arr': profile indexed by the flight's approximate ARRIVAL time (dep + 45min + 0.15 min/mile),
#             capturing congestion at the arrival airport around landing time.
KP_SPEC = [
    ("kp_o_15", "Origin", 15, 3, "dep"),
    ("kp_d_25", "Dest", 25, 3, "dep"),
    ("kp_c_10", "UniqueCarrier", 10, 3, "dep"),
    ("kp_o_45", "Origin", 45, 10, "dep"),
    ("kp_d_8", "Dest", 8, 3, "dep"),
    ("kp_o_8", "Origin", 8, 3, "dep"),
    ("kp_d_60", "Dest", 60, 10, "dep"),
    ("kp_c_40", "UniqueCarrier", 40, 10, "dep"),
    ("kp_da_8", "Dest", 8, 3, "arr"),
    ("kp_da_15", "Dest", 15, 3, "arr"),
    ("kp_oa_15", "Origin", 15, 3, "arr"),
    ("kp_ca_10", "UniqueCarrier", 10, 3, "arr"),
]

# key codings and time-of-day histograms, fit on TRAIN only
key_levels = {col: pd.Index(sorted(train[col].astype(str).unique())) for _, col, _, _, _ in KP_SPEC}
key_maps = {col: {v: i for i, v in enumerate(levels)} for col, levels in key_levels.items()}

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
folds = list(kf.split(np.arange(len(train))))
cell_train = {
    "dep": (minutes_arr(train["DepTime"].to_numpy()) // CELL).astype(int),
    "arr": ((minutes_arr(train["DepTime"].to_numpy()) + 45 + 0.15 * train["Distance"].to_numpy()) % 1440 // CELL).astype(int),
}


def cells_for(df: pd.DataFrame, kind: str) -> np.ndarray:
    if kind == "dep":
        return (minutes_arr(df["DepTime"].to_numpy()) // CELL).astype(int)
    return ((minutes_arr(df["DepTime"].to_numpy()) + 45 + 0.15 * df["Distance"].to_numpy()) % 1440 // CELL).astype(int)


def _hist(keys_code: np.ndarray, n_keys: int, rows: np.ndarray, kind: str) -> tuple:
    cells = cell_train[kind]
    cnt = np.zeros((n_keys, G)); sy = np.zeros((n_keys, G))
    np.add.at(cnt, (keys_code[rows], cells[rows]), 1.0)
    np.add.at(sy, (keys_code[rows], cells[rows]), y_train[rows])
    return cnt, sy

key_codes_tr = {}
hist_full = {}
hist_folds = {}
for col in key_maps:
    codes = train[col].astype(str).map(key_maps[col]).to_numpy()
    key_codes_tr[col] = codes
for kind in ("dep", "arr"):
    for col in key_maps:
        hist_full[(col, kind)] = _hist(key_codes_tr[col], len(key_levels[col]), np.arange(len(train)), kind)
        hist_folds[(col, kind)] = [_hist(key_codes_tr[col], len(key_levels[col]), tr_i, kind) for tr_i, _ in folds]


def _smooth(cnt: np.ndarray, sy: np.ndarray, sigma_min: float, m: float) -> np.ndarray:
    """Circular Gaussian smoothing of the (sum_y, count) curves; shrunk rate per (key, cell)."""
    sc = sigma_min / CELL
    W = max(int(3 * sc), 1)
    off = np.arange(-W, W + 1)
    K = np.exp(-(off ** 2) / (2.0 * sc * sc))
    n = cnt.shape[0]
    smc = np.empty_like(cnt); sms = np.empty_like(cnt)
    for r in range(n):
        smc[r] = np.convolve(np.r_[cnt[r, -W:], cnt[r], cnt[r, :W]], K, mode="same")[W:W + G]
        sms[r] = np.convolve(np.r_[sy[r, -W:], sy[r], sy[r, :W]], K, mode="same")[W:W + G]
    return (sms + m * prior) / (smc + m)

# profile matrices for every spec (full-train)
prof_full = {(name, col, sig, m, kind): _smooth(*hist_full[(col, kind)], sig, m) for name, col, sig, m, kind in KP_SPEC}

# OOF profile values for the TRAINING rows (honest, no self-leakage)
oof_kp = {name: np.full(len(train), prior) for name, _, _, _, _ in KP_SPEC}
for f_idx, (tr_i, va_i) in enumerate(folds):
    for name, col, sig, m, kind in KP_SPEC:
        prof = _smooth(*hist_folds[(col, kind)][f_idx], sig, m)
        codes = key_codes_tr[col][va_i]
        ok = codes >= 0
        oof_kp[name][va_i[ok]] = prof[codes[ok], cell_train[kind][va_i[ok]]]

cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CATS}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering lives here: predict_proba() calls prepare() on unseen rows.
    X = df[RAW_NUM].copy()
    for c in CATS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    cells_by_kind = {kind: cells_for(df, kind) for kind in ("dep", "arr")}
    for name, col, sig, m, kind in KP_SPEC:
        codes = df[col].astype(str).map(key_maps[col]).fillna(-1).astype(int).to_numpy()
        X[name] = np.where(codes >= 0, prof_full[(name, col, sig, m, kind)][np.maximum(codes, 0), cells_by_kind[kind]], prior)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


X_train_full = prepare(train)   # full-train profiles (what unseen rows get)
X_train_fit = X_train_full.copy()
for name, _, _, _, _ in KP_SPEC:
    X_train_fit[name] = oof_kp[name]  # training rows see honest OOF values

# --- model --------------------------------------------------------------------
PARAMS = dict(
    n_estimators=500,
    max_depth=16,
    learning_rate=0.036,
    subsample=0.9,
    colsample_bytree=0.6,
    reg_alpha=0.5,
    tree_method="hist",
    max_bin=128,
    enable_categorical=True,
)

t0 = time.time()
models = []
for s in range(N_SEEDS):
    mdl = xgb.XGBClassifier(**PARAMS, random_state=SEED + s, n_jobs=N_JOBS)
    mdl.fit(X_train_fit, y_train)
    models.append(mdl)
print(f"Tree training: {time.time() - t0:.1f}s ({N_SEEDS} seeds)")

# linear XGBoost on the same features (cat columns as integer codes) -> blend for ranking diversity
X_num_fit = X_train_fit.copy()
for c in CATS:
    X_num_fit[c] = X_train_fit[c].cat.codes.astype(float)
gblinear = xgb.XGBClassifier(booster="gblinear", n_estimators=200, learning_rate=0.1,
                            random_state=SEED, n_jobs=N_JOBS)
gblinear.fit(X_num_fit, y_train)
print(f"Linear training: {time.time() - t0:.1f}s")

GL_W = 0.3  # weight of the linear model in the blended probability


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    p_tree = np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)
    X_num = X.copy()
    for c in CATS:
        X_num[c] = X[c].cat.codes.astype(float)
    p_lin = gblinear.predict_proba(X_num)[:, 1]
    return (1.0 - GL_W) * p_tree + GL_W * p_lin


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
