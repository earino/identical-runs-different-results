"""XGBoost binary classifier for airline delays. THIS IS THE ONLY FILE THE AGENT EDITS.

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

CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().astype(str).unique())) for c in CAT_COLS}


def _tod(df: pd.DataFrame) -> pd.Series:
    return df["DepTime"].fillna(0).astype(int) % 2400


# label-free counts fitted on train (congestion/size proxies)
_tr_tod = _tod(train)
_tr_hour = _tr_tod // 100
CNT_MAPS = {
    "cnt_carrier": train["UniqueCarrier"].astype(str).value_counts().to_dict(),
    "cnt_origin": train["Origin"].astype(str).value_counts().to_dict(),
    "cnt_dest": train["Dest"].astype(str).value_counts().to_dict(),
    "cnt_origin_hour": train["Origin"].astype(str).add("_").add(_tr_hour.astype(str)).value_counts().to_dict(),
    "cnt_dest_hour": train["Dest"].astype(str).add("_").add(_tr_hour.astype(str)).value_counts().to_dict(),
}

# geographic proxies (fit on train)
ORIGIN_MEAN_DIST = train.groupby(train["Origin"].astype(str))["Distance"].mean().to_dict()
DEST_MEAN_DIST = train.groupby(train["Dest"].astype(str))["Distance"].mean().to_dict()
DIST_BINS = np.quantile(train["Distance"].astype(float), np.linspace(0, 1, 9)[1:-1])


def _map_cnt(s: pd.Series, mapping: dict) -> np.ndarray:
    return np.log1p(s.map(mapping).fillna(0).astype(float).to_numpy())


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    # DepTime: scheduled departure hhmm -> time-of-day features
    tod = df["DepTime"].fillna(0).astype(int) % 2400
    hour = tod // 100
    minute = tod % 100
    ang = 2.0 * np.pi * (hour * 60 + minute) / 1440.0
    X["dep_sin"] = np.sin(ang)
    X["dep_cos"] = np.cos(ang)
    X["hour"] = hour
    X["hour_cat"] = pd.Categorical(hour.astype(str), categories=[str(h) for h in range(24)])
    # distance
    dist = df["Distance"].astype(float)
    X["dist"] = dist
    X["dist_log"] = np.log1p(dist)
    om = df["Origin"].astype(str).map(ORIGIN_MEAN_DIST).astype(float)
    dm = df["Dest"].astype(str).map(DEST_MEAN_DIST).astype(float)
    X["origin_meandist"] = om.fillna(dist.mean()).to_numpy()
    X["dest_meandist"] = dm.fillna(dist.mean()).to_numpy()
    X["dist_rel_origin"] = (dist / om.replace(0, np.nan)).clip(0, 5).fillna(1.0).to_numpy()
    X["dist_bucket"] = pd.Categorical(
        np.digitize(dist.to_numpy(), DIST_BINS).astype(str), categories=[str(i) for i in range(8)]
    )
    # congestion / size proxies (counts from train)
    hour_s = tod // 100
    X["cnt_carrier"] = _map_cnt(df["UniqueCarrier"].astype(str), CNT_MAPS["cnt_carrier"])
    X["cnt_origin"] = _map_cnt(df["Origin"].astype(str), CNT_MAPS["cnt_origin"])
    X["cnt_dest"] = _map_cnt(df["Dest"].astype(str), CNT_MAPS["cnt_dest"])
    X["cnt_origin_hour"] = _map_cnt(
        df["Origin"].astype(str).add("_").add(hour_s.astype(str)), CNT_MAPS["cnt_origin_hour"]
    )
    X["cnt_dest_hour"] = _map_cnt(
        df["Dest"].astype(str).add("_").add(hour_s.astype(str)), CNT_MAPS["cnt_dest_hour"]
    )
    # 15-minute departure slot as categorical
    slot = (tod // 15).astype(int)
    X["slot15"] = pd.Categorical(slot.astype(str), categories=[str(s) for s in range(96)])
    # categoricals (levels frozen on train; unseen -> NaN -> xgb missing)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c].astype(str), categories=cat_levels[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
ENSEMBLE = [
    dict(max_depth=10, learning_rate=0.03, seed=42),
    dict(max_depth=8, learning_rate=0.03, seed=7),
]


def make_model(n_estimators: int, p: dict) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=n_estimators,
        subsample=0.7,
        colsample_bytree=0.7,
        colsample_bynode=0.7,
        min_child_weight=20,
        reg_alpha=0.5,
        reg_lambda=2.0,
        max_bin=512,
        tree_method="hist",
        enable_categorical=True,
        n_jobs=N_JOBS,
        **p,
    )


t0 = time.time()
rng = np.random.RandomState(SEED)
idx = rng.permutation(len(train))
n_val = len(train) // 5
val_idx, fit_idx = idx[:n_val], idx[n_val:]
X_fit, y_fit = prepare(train.iloc[fit_idx]), to_y(train.iloc[fit_idx])
X_val, y_val = prepare(train.iloc[val_idx]), to_y(train.iloc[val_idx])
X_tr, y_tr = prepare(train), to_y(train)

models = []
for p in ENSEMBLE:
    es_model = make_model(1600, p)
    es_model.set_params(early_stopping_rounds=50, eval_metric="auc")
    es_model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
    best_n = max(int(es_model.best_iteration) + 1, 30)
    print(f"depth={p['max_depth']} best_iteration={best_n} val_auc={es_model.best_score:.4f}")
    m = make_model(best_n, p)
    m.fit(X_tr, y_tr)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    P = np.mean([m.predict_proba(prepare(df))[:, 1] for m in models], axis=0)
    return P


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
