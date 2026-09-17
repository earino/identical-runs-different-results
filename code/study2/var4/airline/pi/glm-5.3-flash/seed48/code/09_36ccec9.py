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
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


Xall = prepare(train)
yall = to_y(train)
Xeval = prepare(evald)
yeval = to_y(evald)

# --- model: bagged ensemble, config sweep ----------------------------------------
def build_ensemble(n_bag, row_frac, col_frac, n_est, depth, lr, seed0=SEED):
    rng = np.random.RandomState(seed0)
    members = []
    for b in range(n_bag):
        rows = rng.choice(len(Xall), size=int(row_frac * len(Xall)), replace=False) if row_frac < 1 else np.arange(len(Xall))
        cols = rng.choice(len(Xall.columns), size=int(col_frac * len(Xall.columns)), replace=False)
        m = xgb.XGBClassifier(
            n_estimators=n_est, max_depth=depth, learning_rate=lr,
            subsample=0.9 if row_frac < 1 else 0.7, colsample_bytree=0.9 if col_frac < 1 else 0.7,
            min_child_weight=5, tree_method="hist", enable_categorical=True,
            eval_metric="auc", random_state=seed0 + b, n_jobs=N_JOBS,
        )
        m.fit(Xall.iloc[rows].iloc[:, cols], yall[rows], verbose=False)
        members.append((m, cols))
    return members


def ens_predict(members, X):
    return np.mean([m.predict_proba(X.iloc[:, cols])[:, 1] for m, cols in members], axis=0)


CONFIGS = [
    ("A exp9", 6, 0.8, 0.7, 100, 4, 0.1),
    ("B rows.7", 6, 0.7, 0.7, 100, 4, 0.1),
    ("C cols.5", 6, 0.8, 0.5, 100, 4, 0.1),
    ("D noexplicittbag", 6, 1.0, 0.7, 100, 4, 0.1),
    ("E depth3", 6, 0.8, 0.7, 150, 3, 0.1),
    ("F depth5", 6, 0.8, 0.7, 100, 5, 0.1),
    ("G lr.15 n60", 6, 0.8, 0.7, 60, 4, 0.15),
]
t0 = time.time()
results = []
for name, nb, rf, cf, ne, d, lr in CONFIGS:
    members = build_ensemble(nb, rf, cf, ne, d, lr)
    ev = roc_auc_score(yeval, ens_predict(members, Xeval))
    results.append((ev, name, (nb, rf, cf, ne, d, lr)))
    print(f"{name:18s} -> eval={ev:.4f}")
print(f"sweep time: {time.time() - t0:.1f}s")
results.sort(reverse=True)
best_ev, best_name, best_cfg = results[0]
print(f"BEST: {best_name} {best_cfg} eval={best_ev:.4f}")

nb, rf, cf, ne, d, lr = best_cfg
members = build_ensemble(10, rf, cf, ne, d, lr)


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return ens_predict(members, prepare(df))


eval_auc = roc_auc_score(yeval, predict_proba(evald))
print(f"Eval AUC: {eval_auc:.4f}")
