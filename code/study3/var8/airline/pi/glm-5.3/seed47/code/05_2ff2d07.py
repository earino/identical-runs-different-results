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
N_SEEDS = 5

y_train = (train[TARGET] == POSITIVE).astype(int).to_numpy()
prior = float(y_train.mean())

def minutes_arr(dep: np.ndarray) -> np.ndarray:
    """DepTime (hhmm int, with dirty values) -> minutes since midnight, circular."""
    return ((dep // 100) * 60 + (dep % 100)) % 1440

# kernel-smoothed profile specs: (name, key column, sigma minutes, shrinkage m)
KP_SPEC = [
    ("kp_o_15", "Origin", 15, 3),
    ("kp_d_25", "Dest", 25, 3),
    ("kp_c_10", "UniqueCarrier", 10, 3),
    ("kp_o_45", "Origin", 45, 10),
    ("kp_d_8", "Dest", 8, 3),
    ("kp_o_8", "Origin", 8, 3),
    ("kp_d_60", "Dest", 60, 10),
    ("kp_c_40", "UniqueCarrier", 40, 10),
]

# key codings and time-of-day histograms, fit on TRAIN only
key_levels = {col: pd.Index(sorted(train[col].astype(str).unique())) for _, col, _, _ in KP_SPEC}
key_maps = {col: {v: i for i, v in enumerate(levels)} for col, levels in key_levels.items()}

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
folds = list(kf.split(np.arange(len(train))))
cell_train = (minutes_arr(train["DepTime"].to_numpy()) // CELL).astype(int)


def _hist(keys_code: np.ndarray, n_keys: int, rows: np.ndarray) -> tuple:
    cnt = np.zeros((n_keys, G)); sy = np.zeros((n_keys, G))
    np.add.at(cnt, (keys_code[rows], cell_train[rows]), 1.0)
    np.add.at(sy, (keys_code[rows], cell_train[rows]), y_train[rows])
    return cnt, sy

key_codes_tr = {}
hist_full = {}
hist_folds = {}
for col in key_maps:
    codes = train[col].astype(str).map(key_maps[col]).to_numpy()
    key_codes_tr[col] = codes
    hist_full[col] = _hist(codes, len(key_levels[col]), np.arange(len(train)))
    hist_folds[col] = [_hist(codes, len(key_levels[col]), tr_i) for tr_i, _ in folds]


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

# profile matrices for every spec (full-train) and per fold (for OOF training values)
prof_full = {(name, col, sig, m): _smooth(*hist_full[col], sig, m) for name, col, sig, m in KP_SPEC}
prof_folds = {}
for f_idx, (_, va_i) in enumerate(folds):
    pass  # built lazily per fold below

# OOF profile values for the TRAINING rows (honest, no self-leakage)
oof_kp = {name: np.full(len(train), prior) for name, _, _, _ in KP_SPEC}
for f_idx, (tr_i, va_i) in enumerate(folds):
    hist_fold = {col: hist_folds[col][f_idx] for col in hist_folds}
    for name, col, sig, m in KP_SPEC:
        prof = _smooth(*hist_fold[col], sig, m)
        codes = key_codes_tr[col][va_i]
        ok = codes >= 0
        oof_kp[name][va_i[ok]] = prof[codes[ok], cell_train[va_i[ok]]]

cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CATS}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering lives here: predict_proba() calls prepare() on unseen rows.
    X = df[RAW_NUM].copy()
    for c in CATS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    cells = (minutes_arr(df["DepTime"].to_numpy()) // CELL).astype(int)
    for name, col, sig, m in KP_SPEC:
        codes = df[col].astype(str).map(key_maps[col]).fillna(-1).astype(int).to_numpy()
        X[name] = np.where(codes >= 0, prof_full[(name, col, sig, m)][np.maximum(codes, 0), cells], prior)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


X_train_full = prepare(train)   # full-train profiles (what unseen rows get)
X_train_fit = X_train_full.copy()
for name, _, _, _ in KP_SPEC:
    X_train_fit[name] = oof_kp[name]  # training rows see honest OOF values

# --- model --------------------------------------------------------------------
PARAMS = dict(
    n_estimators=600,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.9,
    colsample_bytree=0.6,
    reg_alpha=0.5,
    tree_method="hist",
    enable_categorical=True,
)

t0 = time.time()
models = []
for s in range(N_SEEDS):
    mdl = xgb.XGBClassifier(**PARAMS, random_state=SEED + s, n_jobs=N_JOBS)
    mdl.fit(X_train_fit, y_train)
    models.append(mdl)
print(f"Training time: {time.time() - t0:.1f}s ({N_SEEDS} seeds)")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
