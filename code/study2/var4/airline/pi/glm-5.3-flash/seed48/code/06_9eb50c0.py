"""XGBoost binary classifier for airline delay. Contract: see program.md."""
import json
import os
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

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
# train-only statistics
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
    # counts (train-only stats)
    route = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    X["route_cnt"] = route.map(route_cnt).fillna(0.0).to_numpy()
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


Xall = prepare(train)
yall = to_y(train)
Xeval = prepare(evald)
yeval = to_y(evald)
X_tr, X_val, y_tr, y_val = train_test_split(
    np.arange(len(train)), yall, test_size=0.1, random_state=SEED, stratify=yall
)
Xv, yv = Xall.iloc[X_val], yall[X_val]

model = xgb.XGBClassifier(
    n_estimators=60, max_depth=4, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
    min_child_weight=5, tree_method="hist", enable_categorical=True,
    eval_metric="auc", random_state=SEED, n_jobs=N_JOBS,
)
model.fit(Xall.iloc[X_tr], y_tr, eval_set=[(Xv, yv)], verbose=False)
print(f"val={model.evals_result()['validation_0']['auc'][-1]:.4f}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


eval_auc = roc_auc_score(yeval, predict_proba(evald))
print(f"Eval AUC: {eval_auc:.4f}")
