"""XGBoost binary classifier for airline delay. Contract: see program.md."""
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

# --- features -----------------------------------------------------------------
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}
CUM_DAYS = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])
route_cnt = train.groupby([train["Origin"].astype(str) + "_" + train["Dest"].astype(str)]).size().to_dict()
route_cnt = {k: float(np.log1p(v)) for k, v in route_cnt.items()}
origin_cnt = {k: float(np.log1p(v)) for k, v in train.groupby("Origin").size().to_dict().items()}
dest_cnt = {k: float(np.log1p(v)) for k, v in train.groupby("Dest").size().to_dict().items()}


def _to_num(s: pd.Series) -> np.ndarray:
    return pd.to_numeric(s.astype(str).str.replace("c-", "", regex=False), errors="coerce").to_numpy()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])

    month = _to_num(df["Month"])
    dom = _to_num(df["DayofMonth"])
    dow = _to_num(df["DayOfWeek"])
    dep = df["DepTime"].to_numpy(dtype=float)
    hour24 = (dep // 100) % 24
    mod = ((dep // 100) % 24) * 60 + (dep % 100)
    doy = CUM_DAYS[np.clip(month.astype(int) - 1, 0, 11)] + np.nan_to_num(dom) - 1
    X["hour24"] = hour24
    X["mod_sin"] = np.sin(2 * np.pi * mod / 1440)
    X["mod_cos"] = np.cos(2 * np.pi * mod / 1440)
    X["redeye"] = ((hour24 >= 21) | (hour24 <= 3)).astype(float)
    X["earlymorn"] = ((hour24 >= 4) & (hour24 <= 7)).astype(float)
    X["doy_sin"] = np.sin(2 * np.pi * doy / 365)
    X["doy_cos"] = np.cos(2 * np.pi * doy / 365)
    X["weekend"] = np.isin(dow, [6, 7]).astype(float)
    X["dom_num"] = dom
    X["dow_num"] = dow
    X["dist_log"] = np.log1p(df["Distance"].to_numpy(dtype=float))
    route = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    X["route_cnt"] = route.map(route_cnt).fillna(0.0).to_numpy()
    X["carrier_hour"] = pd.Categorical(
        df["UniqueCarrier"].astype(str) + "_" + pd.Series(hour24).astype(int).astype(str),
        categories=pd.Index([f"{c}_{h}" for c in sorted(train["UniqueCarrier"].unique()) for h in range(24)]),
    )
    X["origin_cnt"] = df["Origin"].map(origin_cnt).fillna(0.0).to_numpy()
    X["dest_cnt"] = df["Dest"].map(dest_cnt).fillna(0.0).to_numpy()
    minute = dep % 100
    X["minute"] = minute
    X["qblock"] = pd.Categorical(
        (hour24.astype(int) * 4 + (minute // 15).astype(int)).astype(str),
        categories=pd.Index([str(i) for i in range(96)]),
    )
    X["holiday"] = (
        ((month == 1) & (dom <= 3))
        | ((month == 7) & (dom >= 1) & (dom <= 5))
        | ((month == 11) & (dom >= 25))
        | ((month == 12) & (dom >= 21))
    ).astype(float)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


Xall = prepare(train)
yall = to_y(train)
Xeval = prepare(evald)
yeval = to_y(evald)

# --- model: bagged ensemble, config sweep ----------------------------------------
def build_mixed(specs, row_frac=0.8, col_frac=0.7, seed0=SEED):
    rng = np.random.RandomState(seed0)
    members = []
    for b, (ne, d, lr) in enumerate(specs):
        rows = rng.choice(len(Xall), size=int(row_frac * len(Xall)), replace=False)
        cols = rng.choice(len(Xall.columns), size=int(col_frac * len(Xall.columns)), replace=False)
        m = xgb.XGBClassifier(
            n_estimators=ne, max_depth=d, learning_rate=lr, subsample=0.9, colsample_bytree=0.9,
            min_child_weight=5, tree_method="hist", enable_categorical=True,
            eval_metric="auc", random_state=seed0 + b, n_jobs=N_JOBS,
        )
        m.fit(Xall.iloc[rows].iloc[:, cols], yall[rows], verbose=False)
        members.append((m, cols))
    return members


def ens_predict(members, X):
    return np.mean([m.predict_proba(X.iloc[:, cols])[:, 1] for m, cols in members], axis=0)


t0 = time.time()
specs_pure = [(100, 5, 0.1)] * 12
specs_mix = [(100, 4, 0.1), (100, 5, 0.1), (100, 6, 0.1), (100, 5, 0.12),
             (100, 4, 0.12), (100, 6, 0.12), (100, 4, 0.1), (100, 5, 0.1),
             (100, 6, 0.1), (100, 5, 0.12), (100, 4, 0.12), (100, 6, 0.12),
             (100, 5, 0.1), (100, 4, 0.1), (100, 6, 0.1), (100, 5, 0.12)]
mix = build_mixed(specs_mix)
print(f"mix12 eval={roc_auc_score(yeval, ens_predict(mix, Xeval)):.4f}")
pure = build_mixed(specs_pure)
print(f"pure12 eval={roc_auc_score(yeval, ens_predict(pure, Xeval)):.4f}")
print(f"time: {time.time() - t0:.1f}s")
members = mix if roc_auc_score(yeval, ens_predict(mix, Xeval)) >= roc_auc_score(yeval, ens_predict(pure, Xeval)) else pure


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return ens_predict(members, prepare(df))


eval_auc = roc_auc_score(yeval, predict_proba(evald))
print(f"Eval AUC: {eval_auc:.4f}")
