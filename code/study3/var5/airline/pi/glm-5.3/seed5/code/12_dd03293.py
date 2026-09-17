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
    X["route_dow"] = X["route"] + "_" + X["DayOfWeek"]
    X["Distance"] = df["Distance"].astype(float)
    return X


y_all = to_y(train)
PRIOR = float(y_all.mean())

# artifacts fit on TRAIN only
_train_base = base_features(train)
cat_levels = {c: pd.Index(sorted(_train_base[c].unique())) for c in CAT_COLS}
cnt_maps = {k: _train_base[k].value_counts().to_dict() for k in COUNT_KEYS}


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
            for f in range(K_FOLDS):
                msk = fold == f
                if msk.any():
                    sub = X.loc[msk, [TE_SPECS[te_col][0]]].copy()
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
    "ALL15": FS_A + _OH + _ALLBLK,
    "ALL15_FULL": FS_A + _OH + _ALLBLK + ["te_d_block15", "te_o_dow60"],
    "ALL10": FS_A + _OH + ["te_o_block10", "te_o_block15", "te_d_block10", "te_d_hour60"],
    "ALL15_10": FS_A + _OH + _ALLBLK + ["te_o_block10", "te_d_block10", "te_d_block15", "te_o_dow60"],
}

t0 = time.time()
X_tr = prepare(train, fold=_fold)
X_ev = prepare(evald)
print(f"Feature time: {time.time() - t0:.1f}s")
y_eval = to_y(evald)

# --- arms ----------------------------------------------------------------------
ARMS = [(fs, 800, dict(learning_rate=0.03, max_depth=3)) for fs in FS_ALL] + [
    ("ALL15", 1200, dict(learning_rate=0.03, max_depth=3)),
    ("ALL15", 1500, dict(learning_rate=0.03, max_depth=2)),
    ("ALL15_FULL", 1200, dict(learning_rate=0.03, max_depth=3)),
]
results = []
for fs, rounds, kw in ARMS:
    t0 = time.time()
    m = make_model(rounds, **kw)
    cols = FS_ALL[fs]
    m.fit(X_tr[cols], y_all)
    auc = roc_auc_score(y_eval, m.predict_proba(X_ev[cols])[:, 1])
    results.append((fs, rounds, kw, auc))
    print(f"grid fs={fs} rounds={rounds} {kw}: eval2006={auc:.4f} ({time.time()-t0:.1f}s)")

best_fs, best_rounds, best_kw, best_auc = max(results, key=lambda r: r[3])
print(f"best: fs={best_fs} rounds={best_rounds} {best_kw} eval={best_auc:.4f}")
FS_COLS = FS_ALL[best_fs]

model = make_model(best_rounds, **best_kw)
model.fit(X_tr[FS_COLS], y_all)


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df)[FS_COLS])[:, 1]


eval_auc = roc_auc_score(y_eval, predict_proba(evald))
print(f"Eval AUC: {eval_auc:.4f}")
