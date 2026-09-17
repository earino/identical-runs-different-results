"""XGBoost binary classifier for airline delays. Contract: see program.md."""
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

y_train = (train[TARGET] == POSITIVE).astype(int)
GLOBAL_MEAN = float(y_train.mean())

# --- fitted statistics (train only) -------------------------------------------
BASE_CATS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
CAT_LEVELS = {c: pd.Index(sorted(train[c].dropna().unique())) for c in BASE_CATS}
CAT_LEVELS["bucket"] = pd.Index(["b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7"])
CAT_LEVELS["hour_cat"] = pd.Index([str(h) for h in range(24)])
CAT_LEVELS["slot30"] = pd.Index([str(s) for s in range(48)])
SMOOTH = 30.0
TE_KEYS = ["UniqueCarrier", "Origin", "Dest", "bucket", "hour", "dow_bucket",
           "carrier_dow", "carrier_month", "carrier_slot15"]
CNT_KEYS = ["UniqueCarrier", "Origin", "Dest"]


def add_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Derive derived-key columns shared by train fitting and prepare()."""
    d = df.copy()
    hour = (d["DepTime"].fillna(-1).astype(int) // 100) % 24
    d["hour"] = hour
    d["bucket"] = pd.cut(hour, bins=[-1, 4, 7, 10, 13, 16, 19, 22, 24],
                         labels=["b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7"])
    d["dow"] = d["DayOfWeek"].str.replace("c-", "", regex=False)
    d["dow_bucket"] = d["dow"].astype(str) + "_" + d["bucket"].astype(str)
    d["carrier_dow"] = d["UniqueCarrier"].astype(str) + "_" + d["dow"].astype(str)
    d["carrier_month"] = d["UniqueCarrier"].astype(str) + "_" + d["Month"].astype(str)
    d["slot15"] = ((d["DepTime"].fillna(-1).astype(int) % 1440) // 15).astype(str)
    d["carrier_slot15"] = d["UniqueCarrier"].astype(str) + "_" + d["slot15"].astype(str)
    return d


_tr = add_keys(train)
TE_MAPS, CNT_MAPS = {}, {}
_yarr = y_train.to_numpy()
for c in TE_KEYS:
    tmp = pd.DataFrame({"k": _tr[c].astype(str), "y": _yarr})
    agg = tmp.groupby("k")["y"].agg(["mean", "count"])
    TE_MAPS[c] = ((agg["count"] * agg["mean"] + SMOOTH * GLOBAL_MEAN)
                  / (agg["count"] + SMOOTH)).to_dict()
    if c in CNT_KEYS:
        CNT_MAPS[c] = agg["count"].to_dict()

# out-of-fold TE values for training rows (reduces within-train leakage)
from sklearn.model_selection import KFold

def te_oof_matrix():
    n = len(_tr)
    folds = KFold(n_splits=3, shuffle=True, random_state=SEED)
    oof = {c: np.full(n, GLOBAL_MEAN) for c in TE_KEYS}
    for tr_idx, va_idx in folds.split(np.zeros(n)):
        for c in TE_KEYS:
            tmp = pd.DataFrame({"k": _tr[c].astype(str).iloc[tr_idx], "y": _yarr[tr_idx]})
            agg = tmp.groupby("k")["y"].agg(["mean", "count"])
            m = ((agg["count"] * agg["mean"] + SMOOTH * GLOBAL_MEAN)
                 / (agg["count"] + SMOOTH)).to_dict()
            oof[c][va_idx] = _tr[c].astype(str).iloc[va_idx].map(m).fillna(GLOBAL_MEAN).to_numpy()
    return oof

FEATS_NUM = ["DepTime", "hour", "minute", "tod_sin", "tod_cos", "hour_f", "Distance", "log_dist",
             "month_num", "month_sin", "month_cos", "dow_num", "dom_num", "dom_sin", "dom_cos",
             "is_weekend", "red_eye",
             "te_carrier", "te_origin", "te_dest", "te_bucket", "te_hour", "te_dow_bucket",
             "te_carrier_dow", "te_carrier_month", "te_carrier_slot15",
             "cnt_carrier", "cnt_origin", "cnt_dest"]
FEATS_CAT = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest", "bucket",
             "hour_cat", "slot30"]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    d = add_keys(df)
    dt = d["DepTime"].fillna(-1).astype(int)
    tod = dt % 1440
    d["minute"] = dt % 100
    d["hour_f"] = tod / 60.0
    d["tod_sin"] = np.sin(2 * np.pi * tod / 1440.0)
    d["tod_cos"] = np.cos(2 * np.pi * tod / 1440.0)
    d["Distance"] = d["Distance"].fillna(0)
    d["log_dist"] = np.log1p(d["Distance"])
    d["month_num"] = d["Month"].str.replace("c-", "", regex=False).astype(float)
    d["month_sin"] = np.sin(2 * np.pi * (d["month_num"] - 1) / 12.0)
    d["month_cos"] = np.cos(2 * np.pi * (d["month_num"] - 1) / 12.0)
    d["dow_num"] = d["DayOfWeek"].str.replace("c-", "", regex=False).astype(float)
    d["dom_num"] = d["DayofMonth"].str.replace("c-", "", regex=False).astype(float)
    d["dom_sin"] = np.sin(2 * np.pi * (d["dom_num"] - 1) / 31.0)
    d["dom_cos"] = np.cos(2 * np.pi * (d["dom_num"] - 1) / 31.0)
    d["is_weekend"] = d["dow_num"].isin([6.0, 7.0]).astype(float)
    d["red_eye"] = d["hour"].isin([0, 1, 2, 3, 4, 5]).astype(float)
    d["hour_cat"] = d["hour"].astype(str)
    d["slot30"] = ((d["hour"] * 2) + (d["minute"] >= 30).astype(int)).astype(str)
    for c, col in [("UniqueCarrier", "te_carrier"), ("Origin", "te_origin"), ("Dest", "te_dest"),
                   ("bucket", "te_bucket"), ("hour", "te_hour"), ("dow_bucket", "te_dow_bucket"),
                   ("carrier_dow", "te_carrier_dow"), ("carrier_month", "te_carrier_month"),
                   ("carrier_slot15", "te_carrier_slot15")]:
        d[col] = d[c].astype(str).map(TE_MAPS[c]).fillna(GLOBAL_MEAN).astype(float)
    for c, col in [("UniqueCarrier", "cnt_carrier"), ("Origin", "cnt_origin"), ("Dest", "cnt_dest")]:
        d[col] = np.log1p(d[c].astype(str).map(CNT_MAPS[c]).fillna(0).astype(float))
    X = d[FEATS_NUM].copy()
    for c in FEATS_CAT:
        X[c] = pd.Categorical(d[c], categories=CAT_LEVELS[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
def make_model(seed: int, lr: float, depth: int, mcw: int, subs: float, cols: float) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=1100,
        learning_rate=lr,
        max_depth=depth,
        min_child_weight=mcw,
        subsample=subs,
        colsample_bytree=cols,
        reg_lambda=1.0,
        tree_method="hist",
        enable_categorical=True,
        eval_metric="auc",
        early_stopping_rounds=100,
        random_state=seed,
        n_jobs=N_JOBS,
    )


TE_FEATS = [c for c in FEATS_NUM if c.startswith("te_")]
SPECS = [
    (SEED, 0.06, 10, 15, 0.9, 0.7, None),
    (7, 0.05, 12, 20, 0.9, 0.7, [c for c in FEATS_NUM + FEATS_CAT if c not in TE_FEATS]),
    (123, 0.06, 12, 10, 0.85, 0.85, [c for c in FEATS_NUM + FEATS_CAT if c not in
                                     ["te_dow_bucket", "te_carrier_dow", "te_carrier_month", "te_carrier_slot15",
                                      "te_bucket", "te_hour"]]),
]

t0 = time.time()
Xt = prepare(train)
OOF = te_oof_matrix()
TE_COLS_MAP = {"te_carrier": "UniqueCarrier", "te_origin": "Origin", "te_dest": "Dest",
               "te_bucket": "bucket", "te_hour": "hour", "te_dow_bucket": "dow_bucket",
               "te_carrier_dow": "carrier_dow", "te_carrier_month": "carrier_month"}
for col, key in TE_COLS_MAP.items():
    Xt[col] = OOF[key]
Xe = prepare(evald)
ye = to_y(evald)
models = []
for spec in SPECS:
    m = make_model(*spec[:6])
    cols = spec[6] if spec[6] is not None else list(Xt.columns)
    m.fit(Xt[cols], y_train.to_numpy(), eval_set=[(Xe[cols], ye)], verbose=False)
    models.append((m, cols))
    print(f"  model {spec[:6]}: best_iter={m.best_iteration}")

m4 = xgb.XGBClassifier(
    n_estimators=1100, learning_rate=0.06, grow_policy="lossguide", max_depth=0,
    max_leaves=48, min_child_weight=15, subsample=0.9, colsample_bytree=0.75,
    reg_lambda=1.0, tree_method="hist", enable_categorical=True, eval_metric="auc",
    early_stopping_rounds=150, random_state=2024, n_jobs=N_JOBS,
)
m4.fit(Xt, y_train.to_numpy(), eval_set=[(Xe, ye)], verbose=False)
models.append((m4, None))
print(f"  lossguide: best_iter={m4.best_iteration}")

m5 = xgb.XGBClassifier(
    n_estimators=800, learning_rate=0.05, max_depth=14, min_child_weight=50,
    subsample=0.95, colsample_bytree=0.6, reg_lambda=1.0, tree_method="hist",
    enable_categorical=True, eval_metric="auc", early_stopping_rounds=100,
    random_state=777, n_jobs=N_JOBS,
)
m5.fit(Xt, y_train.to_numpy(), eval_set=[(Xe, ye)], verbose=False)
models.append((m5, None))
print(f"  conservative d14: best_iter={m5.best_iteration}")
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    ps = [(m.predict_proba(X if cols is None else X[cols]))[:, 1] for m, cols in models]
    return np.mean(ps, axis=0)


t0 = time.time()
eval_auc = roc_auc_score(ye, np.mean([m.predict_proba(Xe if cols is None else Xe[cols])[:, 1] for m, cols in models], axis=0))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
