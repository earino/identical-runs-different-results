"""XGBoost binary classifier for airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Method (chosen over 37 tracked experiments, eval AUC 0.7141 -> 0.7334):
  - Features: raw schedule columns plus out-of-fold target encodings (TE) of congestion keys
    origin/dest x hour / 15-min / 30-min departure block, origin x day-of-week, plus a
    hand-built label-free holiday-distance signal. All encodings are fit on TRAIN only;
    training rows use 5-fold out-of-fold TE values to avoid leakage, prediction rows use
    full-train statistics.
  - Model: mean of 5 shallow XGBoost classifiers (depth 3, lr 0.03, 1200 rounds,
    L2 10, L1 1, per-node/per-tree column subsampling, seeds 0-4). Shallow + slow +
    smoothed TEs were what generalized across the 2005 -> 2006 shift.
"""
import json
import os

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

TASK = json.load(open("task.json"))
TARGET = TASK["target"]
POSITIVE = TASK["positive_label"]
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42
K_FOLDS = 5

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- feature engineering -------------------------------------------------------
CAT_COLS = ["UniqueCarrier", "Origin", "Dest", "Month", "DayofMonth", "DayOfWeek"]
# congestion keys -> smoothing prior counts m
TE_SPECS = {
    "te_o_hour60": ("o_hour", 60),
    "te_o_block15": ("o_block15", 80),
    "te_o_block30": ("o_block30", 80),
    "te_d_hour60": ("d_hour", 60),
    "te_d_block30": ("d_block30", 80),
    "te_d_block15": ("d_block15", 80),
    "te_o_dow60": ("o_dow", 60),
}


def base_features(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["DepTime"] = df["DepTime"].astype(float)
    X["UniqueCarrier"] = df["UniqueCarrier"].astype(str)
    X["Origin"] = df["Origin"].astype(str)
    X["Dest"] = df["Dest"].astype(str)
    X["Month"] = df["Month"].astype(str)
    X["DayofMonth"] = df["DayofMonth"].astype(str)
    X["DayOfWeek"] = df["DayOfWeek"].astype(str)
    hour = (df["DepTime"] // 100) % 24  # DepTime can exceed 2400 (after-midnight)
    dep_min = hour * 60 + df["DepTime"] % 100
    X["o_hour"] = X["Origin"] + "_" + hour.astype(str)
    X["d_hour"] = X["Dest"] + "_" + hour.astype(str)
    X["o_block15"] = X["Origin"] + "_" + (dep_min // 15).astype(int).astype(str)
    X["o_block30"] = X["Origin"] + "_" + (dep_min // 30).astype(int).astype(str)
    X["d_block15"] = X["Dest"] + "_" + (dep_min // 15).astype(int).astype(str)
    X["d_block30"] = X["Dest"] + "_" + (dep_min // 30).astype(int).astype(str)
    X["o_dow"] = X["Origin"] + "_" + X["DayOfWeek"]
    X["Distance"] = df["Distance"].astype(float)
    # label-free calendar signal: circular distance (days) to nearest major US holiday
    _md = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    mo = df["Month"].str.replace("c-", "", regex=False).astype(int).to_numpy()
    dy = df["DayofMonth"].str.replace("c-", "", regex=False).astype(int).to_numpy()
    doy = np.array([_md[m - 1] for m in mo]) + dy
    hol = np.array([1, 148, 185, 247, 329, 359])  # Jan1, May28, Jul4, Sep4, Nov25, Dec25
    d = np.abs(doy[:, None] - hol[None, :])
    d = np.minimum(d, 365 - d)
    X["hol_dist"] = d.min(axis=1)
    X["hol_near"] = (X["hol_dist"] <= 2).astype(float)
    return X


y_all = to_y(train)
PRIOR = float(y_all.mean())

# artifacts fit on TRAIN only
_train_base = base_features(train)
cat_levels = {c: pd.Index(sorted(_train_base[c].unique())) for c in CAT_COLS}


def build_stats(mask: np.ndarray) -> dict:
    stats = {}
    for te_col, (key_col, m) in TE_SPECS.items():
        g = pd.DataFrame({"k": _train_base[key_col][mask], "y": y_all[mask]}).groupby("k")["y"]
        stats[te_col] = (g.sum().to_dict(), g.count().to_dict())
    return stats


_fold = np.random.RandomState(SEED).randint(0, K_FOLDS, len(train))
FULL_STATS = build_stats(np.ones(len(train), dtype=bool))
FOLD_STATS = [build_stats(_fold != f) for f in range(K_FOLDS)]


def attach_te(X: pd.DataFrame, stats: dict, cols: list) -> None:
    for te_col in cols:
        key_col, m = TE_SPECS[te_col]
        sum_map, cnt_map = stats[te_col]
        s = X[key_col].map(sum_map).astype(float).to_numpy()
        k = X[key_col].map(cnt_map).astype(float).to_numpy()
        X[te_col] = np.where(k > 0, (s + m * PRIOR) / (k + m), PRIOR)


def prepare(df: pd.DataFrame, fold: np.ndarray | None = None) -> pd.DataFrame:
    X = base_features(df)
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    te_cols = list(TE_SPECS)
    if fold is None:
        attach_te(X, FULL_STATS, te_cols)
    else:  # out-of-fold TE for training rows (leak-free)
        for te_col in te_cols:
            X[te_col] = np.nan
            key_col = TE_SPECS[te_col][0]
            for f in range(K_FOLDS):
                msk = fold == f
                if msk.any():
                    sub = X.loc[msk, [key_col]].copy()
                    attach_te(sub, FOLD_STATS[f], [te_col])
                    X.loc[msk, te_col] = sub[te_col].to_numpy()
    X["o_x_d"] = X["te_o_block15"] * X["te_d_block30"]  # joint origin/dest congestion stress
    return X


# NOTE: exact column order matters for the column-subsampling ensemble (kept from the best run)
FS_COLS = [
    "DepTime", "Distance", "Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest",
    "te_o_hour60", "te_o_block15", "te_o_block30", "te_d_block30", "te_d_hour60", "te_d_block15",
    "te_o_dow60", "hol_dist", "hol_near", "o_x_d",
]

# --- model --------------------------------------------------------------------
X_tr = prepare(train, fold=_fold)
X_ev = prepare(evald)
y_eval = to_y(evald)

models = []
for i in range(5):
    m = xgb.XGBClassifier(
        n_estimators=1200,
        max_depth=3,
        learning_rate=0.03,
        reg_lambda=10,
        reg_alpha=1,
        colsample_bynode=0.8,
        colsample_bytree=0.9,
        tree_method="hist",
        enable_categorical=True,
        random_state=i,
        n_jobs=N_JOBS,
    )
    m.fit(X_tr[FS_COLS], y_all)
    models.append(m)

p_ev = np.mean([m.predict_proba(X_ev[FS_COLS])[:, 1] for m in models], axis=0)
print(f"Eval AUC: {roc_auc_score(y_eval, p_ev):.4f}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)[FS_COLS]
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)
