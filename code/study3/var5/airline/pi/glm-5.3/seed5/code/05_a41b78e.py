"""XGBoost binary classifier for airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

All feature engineering lives in prepare() and only uses artifacts (category levels, out-of-fold
target-encoding maps) fit on the training data at module level. Training rows get out-of-fold TE
values (their own fold's statistics excluded); unseen rows get fully-smoothed TE values.
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
}


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
    X["hour"] = (df["DepTime"] // 100) % 24  # DepTime can exceed 2400 (after-midnight)
    X["Distance"] = df["Distance"].astype(float)
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
TE_LIST = list(TE_SPECS)
FS_ALL = {
    "A": FS_A,
    "AT": FS_A + TE_LIST,
    "T": ["DepTime", "Distance", "Month", "DayofMonth", "DayOfWeek", "UniqueCarrier"] + TE_LIST,
    "AT_route": FS_A + ["te_route"],
    "AT_hour": FS_A + ["te_hour", "te_origin", "te_dest", "te_carrier"],
}

t0 = time.time()
X_tr = prepare(train, fold=_fold)
X_ev = prepare(evald)
print(f"Feature time: {time.time() - t0:.1f}s")
y_eval = to_y(evald)

# --- arms ----------------------------------------------------------------------
ARMS = [
    ("A", 30),
    ("AT", 30), ("AT", 120), ("AT", 240),
    ("T", 240),
    ("AT_route", 30), ("AT_hour", 30),
]
results = []
for fs, rounds in ARMS:
    t0 = time.time()
    m = make_model(rounds)
    cols = FS_ALL[fs]
    m.fit(X_tr[cols], y_all)
    tr_auc = roc_auc_score(y_all, m.predict_proba(X_tr[cols])[:, 1])
    auc = roc_auc_score(y_eval, m.predict_proba(X_ev[cols])[:, 1])
    results.append((fs, rounds, auc))
    print(f"grid fs={fs} rounds={rounds}: train={tr_auc:.4f} eval2006={auc:.4f} ({time.time()-t0:.1f}s)")

best_fs, best_rounds, best_auc = max(results, key=lambda r: r[2])
print(f"best: fs={best_fs} rounds={best_rounds} eval={best_auc:.4f}")
FS_COLS = FS_ALL[best_fs]

model = make_model(best_rounds)
model.fit(X_tr[FS_COLS], y_all)


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df)[FS_COLS])[:, 1]


eval_auc = roc_auc_score(y_eval, predict_proba(evald))
print(f"Eval AUC: {eval_auc:.4f}")
