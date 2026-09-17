"""XGBoost binary classifier for airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

All feature engineering lives in prepare() and only uses artifacts (category levels, out-of-fold
target-encoding maps, train count tables) fit on the training data at module level.
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
K_FOLDS = 5

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- feature engineering -------------------------------------------------------
CAT_COLS = ["UniqueCarrier", "Origin", "Dest", "Month", "DayofMonth", "DayOfWeek"]
TE_SPECS = {
    "te_route": ("route", 25),
    "te_origin": ("Origin", 15),
    "te_dest": ("Dest", 15),
    "te_hour": ("hour", 100),
    "te_carrier": ("UniqueCarrier", 50),
    "te_month": ("Month", 50),
    "te_dow": ("DayOfWeek", 50),
    "te_o_hour60": ("o_hour", 60),
    "te_o_hour150": ("o_hour", 150),
    "te_d_hour60": ("d_hour", 60),
    "te_block100": ("block", 100),
    "te_c_hour60": ("c_hour", 60),
    "te_o_month60": ("o_month", 60),
    "te_o_block30": ("o_block30", 80),
    "te_o_block15": ("o_block15", 80),
    "te_o_block30m50": ("o_block30", 50),
    "te_d_block30": ("d_block30", 80),
    "te_d_block15": ("d_block15", 80),
    "te_o_dow60": ("o_dow", 60),
    "te_o_block10": ("o_block10", 60),
    "te_d_block10": ("d_block10", 60),
    "te_o_block15h": ("o_block15", 40, "o_hour", 60),
    "te_d_block15h": ("d_block15", 40, "d_hour", 60),
    "te_o_block30h": ("o_block30", 40, "o_hour", 60),
    "te_d_block30h": ("d_block30", 40, "d_hour", 60),
    "te_dow_hour": ("dow_hour", 60),
    "te_o_week": ("o_week", 80),
}
COUNT_KEYS = ["o_hour", "o_dow", "route_dow"]  # label-free volume features


def base_features(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["DepTime"] = df["DepTime"].astype(float)
    X["UniqueCarrier"] = df["UniqueCarrier"].astype(str)
    X["Origin"] = df["Origin"].astype(str)
    X["Dest"] = df["Dest"].astype(str)
    X["route"] = X["Origin"] + "-" + X["Dest"]
    X["Month"] = df["Month"].astype(str)
    X["DayofMonth"] = df["DayofMonth"].astype(str)
    X["DayOfWeek"] = df["DayOfWeek"].astype(str)
    hour = (df["DepTime"] // 100) % 24  # DepTime can exceed 2400 (after-midnight)
    X["hour"] = hour
    dep_min = hour * 60 + df["DepTime"] % 100
    X["block"] = (dep_min // 15).astype(int)
    X["o_hour"] = X["Origin"] + "_" + hour.astype(str)
    X["c_hour"] = X["UniqueCarrier"] + "_" + hour.astype(str)
    X["o_month"] = X["Origin"] + "_" + X["Month"]
    X["o_block30"] = X["Origin"] + "_" + (dep_min // 30).astype(int).astype(str)
    X["o_block15"] = X["Origin"] + "_" + (dep_min // 15).astype(int).astype(str)
    X["d_block30"] = X["Dest"] + "_" + (dep_min // 30).astype(int).astype(str)
    X["d_block15"] = X["Dest"] + "_" + (dep_min // 15).astype(int).astype(str)
    X["o_block10"] = X["Origin"] + "_" + (dep_min // 10).astype(int).astype(str)
    X["d_block10"] = X["Dest"] + "_" + (dep_min // 10).astype(int).astype(str)
    X["d_hour"] = X["Dest"] + "_" + hour.astype(str)
    X["o_dow"] = X["Origin"] + "_" + X["DayOfWeek"]
    X["dow_hour"] = X["DayOfWeek"] + "_" + hour.astype(str)
    X["route_dow"] = X["route"] + "_" + X["DayOfWeek"]
    X["Distance"] = df["Distance"].astype(float)
    # distance (days) to nearest major US holiday window - fixed by month/day, transfers across years
    _md = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    mo = df["Month"].str.replace("c-", "", regex=False).astype(int).to_numpy()
    dy = df["DayofMonth"].str.replace("c-", "", regex=False).astype(int).to_numpy()
    doy = np.array([_md[m - 1] for m in mo]) + dy
    hol = np.array([1, 148, 185, 247, 329, 359])  # Jan1, May28, Jul4, Sep4, Nov25, Dec25
    d = np.abs(doy[:, None] - hol[None, :])
    d = np.minimum(d, 365 - d)
    X["hol_dist"] = d.min(axis=1)
    X["hol_near"] = (X["hol_dist"] <= 2).astype(float)
    X["sin_doy"] = np.sin(2 * np.pi * doy / 365.0)
    X["cos_doy"] = np.cos(2 * np.pi * doy / 365.0)
    X["o_week"] = X["Origin"] + "_" + (doy // 7).astype(str)
    return X


y_all = to_y(train)
PRIOR = float(y_all.mean())

# artifacts fit on TRAIN only
_train_base = base_features(train)
cat_levels = {c: pd.Index(sorted(_train_base[c].unique())) for c in CAT_COLS}
cnt_maps = {k: _train_base[k].value_counts().to_dict() for k in COUNT_KEYS}


def build_stats(mask: np.ndarray) -> dict:
    stats = {}
    for te_col, spec in TE_SPECS.items():
        key_col = spec[0]
        g = pd.DataFrame({"k": _train_base[key_col][mask], "y": y_all[mask]}).groupby("k")["y"]
        stats[te_col] = (g.sum().to_dict(), g.count().to_dict())
    return stats


def key_stats(stats: dict, key_col: str):
    """(sum_map, cnt_map) for a raw key, taken from any TE spec using that key."""
    for te_col, spec in TE_SPECS.items():
        if spec[0] == key_col:
            return stats[te_col]
    raise KeyError(key_col)


_fold = np.random.RandomState(SEED).randint(0, K_FOLDS, len(train))
FULL_STATS = build_stats(np.ones(len(train), dtype=bool))
FOLD_STATS = [build_stats(_fold != f) for f in range(K_FOLDS)]


def attach_te(X: pd.DataFrame, stats: dict, cols: list) -> None:
    for te_col in cols:
        spec = TE_SPECS[te_col]
        key_col, m = spec[0], spec[1]
        sum_map, cnt_map = stats[te_col]
        s = X[key_col].map(sum_map).astype(float).to_numpy()
        k = X[key_col].map(cnt_map).astype(float).to_numpy()
        if len(spec) >= 4:  # hierarchical: shrink toward parent-key TE, not the global prior
            pkey, pm = spec[2], spec[3]
            psum_map, pcnt_map = key_stats(stats, pkey)
            ps = X[pkey].map(psum_map).astype(float).to_numpy()
            pk = X[pkey].map(pcnt_map).astype(float).to_numpy()
            parent = np.where(pk > 0, (ps + pm * PRIOR) / (pk + pm), PRIOR)
            X[te_col] = np.where(k > 0, (s + m * parent) / (k + m), parent)
        else:
            X[te_col] = np.where(k > 0, (s + m * PRIOR) / (k + m), PRIOR)


def attach_counts(X: pd.DataFrame) -> None:
    for k in COUNT_KEYS:
        X["n_" + k] = np.log1p(X[k].map(cnt_maps[k]).fillna(0).astype(float))


def prepare(df: pd.DataFrame, fold: np.ndarray | None = None) -> pd.DataFrame:
    X = base_features(df)
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    attach_counts(X)
    te_cols = list(TE_SPECS)
    if fold is None:
        attach_te(X, FULL_STATS, te_cols)
    else:  # out-of-fold TE for training rows
        for te_col in te_cols:
            X[te_col] = np.nan
            spec = TE_SPECS[te_col]
            need_cols = [spec[0]] + ([spec[2]] if len(spec) >= 4 else [])
            for f in range(K_FOLDS):
                msk = fold == f
                if msk.any():
                    sub = X.loc[msk, need_cols].copy()
                    attach_te(sub, FOLD_STATS[f], [te_col])
                    X.loc[msk, te_col] = sub[te_col].to_numpy()
    return X


# --- model --------------------------------------------------------------------
def make_model(n_estimators: int, **kw) -> xgb.XGBClassifier:
    params = dict(
        n_estimators=n_estimators,
        max_depth=6,
        learning_rate=0.1,
        tree_method="hist",
        enable_categorical=True,
        random_state=SEED,
        n_jobs=N_JOBS,
    )
    params.update(kw)
    return xgb.XGBClassifier(**params)


FS_A = ["DepTime", "Distance", "Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
CNT = ["n_" + k for k in COUNT_KEYS]
_OH = ["te_o_hour60"]
_ALLBLK = ["te_o_block15", "te_o_block30", "te_d_block30", "te_d_hour60"]
FS_ALL = {
    "ALL15_FULL": FS_A + _OH + _ALLBLK + ["te_d_block15", "te_o_dow60"],
    "ALL15_DH2": FS_A + _OH + _ALLBLK + ["te_d_block15", "te_o_dow60", "te_dow_hour"],
    "HIER": FS_A + _OH + ["te_o_block15h", "te_o_block30", "te_d_block30h", "te_d_block15h", "te_d_hour60", "te_o_dow60"],
    "HIER_ALL": FS_A + _OH + ["te_o_block15h", "te_o_block30h", "te_o_block30", "te_d_block30h", "te_d_block15h", "te_d_hour60", "te_o_dow60"],
    "ALL15_HOL": FS_A + _OH + _ALLBLK + ["te_d_block15", "te_o_dow60", "hol_dist", "hol_near"],
    "HOL_WEEK": FS_A + _OH + _ALLBLK + ["te_d_block15", "te_o_dow60", "hol_dist", "hol_near", "te_o_week"],
}

t0 = time.time()
X_tr = prepare(train, fold=_fold)
X_ev = prepare(evald)
print(f"Feature time: {time.time() - t0:.1f}s")
y_eval = to_y(evald)

# --- arms ----------------------------------------------------------------------
REC = dict(learning_rate=0.03, max_depth=3, n_estimators=1200, reg_lambda=10)
COLS = FS_ALL["ALL15_FULL"]
COLS_HOL = FS_ALL["ALL15_HOL"]
COLS_WEEK = FS_ALL["HOL_WEEK"]

def fit1(rounds, cols=None, **kw):
    m = make_model(rounds, **{k: v for k, v in kw.items()})
    m.fit(X_tr[cols or COLS], y_all)
    return m

def fit1_oof(rounds, fold_seed, cols=None, k=K_FOLDS, **kw):
    """Member trained on its own OOF-TE frame (fold_seed decorrelates the TE features)."""
    fold = np.random.RandomState(fold_seed).randint(0, k, len(train))
    stats = [build_stats(fold != f) for f in range(k)]
    X = base_features(train)
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])
    attach_counts(X)
    for te_col in TE_SPECS:
        X[te_col] = np.nan
        for f in range(k):
            msk = fold == f
            if msk.any():
                spec = TE_SPECS[te_col]
                need = [spec[0]] + ([spec[2]] if len(spec) >= 4 else [])
                sub = X.loc[msk, need].copy()
                attach_te(sub, stats[f], [te_col])
                X.loc[msk, te_col] = sub[te_col].to_numpy()
    m = make_model(rounds, **kw)
    m.fit(X[cols or COLS], y_all)
    return m

BASE = dict(learning_rate=0.03, max_depth=3, reg_lambda=10, reg_alpha=1)
ARMS = {
    "hol_ens5": lambda: [fit1(1200, cols=COLS_HOL, colsample_bytree=0.85, random_state=i, **BASE) for i in range(5)],
    "week_single": lambda: [fit1(1200, cols=COLS_WEEK, **BASE)],
    "week_ens5": lambda: [fit1(1200, cols=COLS_WEEK, colsample_bytree=0.85, random_state=i, **BASE) for i in range(5)],
    "k10_ens5": lambda: [fit1_oof(1200, fold_seed=200 + i, k=10, cols=COLS_HOL, colsample_bytree=0.85, random_state=i, **BASE) for i in range(5)],
    "bynode_ens5": lambda: [fit1(1200, cols=COLS_HOL, colsample_bynode=0.8, random_state=i, **BASE) for i in range(5)],
    "ss95_ens5": lambda: [fit1(1200, cols=COLS_HOL, colsample_bytree=0.85, subsample=0.95, random_state=i, **BASE) for i in range(5)],
}

results = []
for name, builder in ARMS.items():
    t0 = time.time()
    models = builder()
    cols_used = COLS_WEEK if name.startswith("week") else COLS_HOL
    p = np.mean([m.predict_proba(X_ev[cols_used])[:, 1] for m in models], axis=0)
    auc = roc_auc_score(y_eval, p)
    results.append((name, models, cols_used, auc))
    print(f"grid {name}: eval2006={auc:.4f} ({time.time()-t0:.1f}s)")

best_name, models, FS_COLS, best_auc = max(results, key=lambda r: r[3])
print(f"best: {best_name} eval={best_auc:.4f}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)[FS_COLS]
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


eval_auc = roc_auc_score(y_eval, predict_proba(evald))
print(f"Eval AUC: {eval_auc:.4f}")
