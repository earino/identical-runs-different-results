"""XGBoost airline delay classifier with cross-fitted target encoding + shallow ensemble."""
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

# --- base features ------------------------------------------------------------
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


y_train = to_y(train)
prior = float(y_train.mean())

# --- target encoding keys (fit on training data only) -------------------------
SMOOTH = 20.0
TE_COLS = ["Origin", "Dest", "UniqueCarrier", "route", "Origin_hour", "Dest_hour", "Carrier_hour", "route_hour", "carrier_route", "route_bin", "route_bin15", "Origin_bin", "Dest_bin", "Carrier_bin", "route_bin10", "Origin_bin15", "Dest_bin15", "Carrier_bin15", "hour_dow", "Origin_dow", "Dest_dow", "Carrier_dow", "route_bin5", "Origin_bin10", "Dest_bin10", "Carrier_bin10", "Origin_bin5", "Dest_bin5", "Carrier_bin5", "carrier_origin", "carrier_dest", "route_dow", "carrier_route_bin", "carrier_route_bin15", "carrier_origin_bin", "carrier_dest_bin", "carrier_route_bin5", "carrier_origin_bin15", "carrier_dest_bin15"]


def make_keys(df: pd.DataFrame) -> dict:
    dept = pd.to_numeric(df["DepTime"], errors="coerce")
    hour = ((dept // 100) % 24).astype(int).astype(str)
    tmin = (dept // 100) * 60 + (dept % 100)
    bin30 = (tmin // 30).astype(int).astype(str)
    bin15 = (tmin // 15).astype(int).astype(str)
    bin10 = (tmin // 10).astype(int).astype(str)
    bin5 = (tmin // 5).astype(int).astype(str)
    return {
        "Origin": df["Origin"].astype(str),
        "Dest": df["Dest"].astype(str),
        "UniqueCarrier": df["UniqueCarrier"].astype(str),
        "route": df["Origin"].astype(str) + "_" + df["Dest"].astype(str),
        "Origin_hour": df["Origin"].astype(str) + "_" + hour,
        "Dest_hour": df["Dest"].astype(str) + "_" + hour,
        "Carrier_hour": df["UniqueCarrier"].astype(str) + "_" + hour,
        "route_hour": df["Origin"].astype(str) + "_" + df["Dest"].astype(str) + "_" + hour,
        "carrier_route": df["UniqueCarrier"].astype(str) + "_" + df["Origin"].astype(str) + "_" + df["Dest"].astype(str),
        "route_bin": df["Origin"].astype(str) + "_" + df["Dest"].astype(str) + "_" + bin30,
        "route_bin15": df["Origin"].astype(str) + "_" + df["Dest"].astype(str) + "_" + bin15,
        "Origin_bin": df["Origin"].astype(str) + "_" + bin30,
        "Dest_bin": df["Dest"].astype(str) + "_" + bin30,
        "Carrier_bin": df["UniqueCarrier"].astype(str) + "_" + bin30,
        "route_bin10": df["Origin"].astype(str) + "_" + df["Dest"].astype(str) + "_" + bin10,
        "Origin_bin15": df["Origin"].astype(str) + "_" + bin15,
        "Dest_bin15": df["Dest"].astype(str) + "_" + bin15,
        "Carrier_bin15": df["UniqueCarrier"].astype(str) + "_" + bin15,
        "hour_dow": hour + "_" + df["DayOfWeek"].astype(str),
        "Origin_dow": df["Origin"].astype(str) + "_" + df["DayOfWeek"].astype(str),
        "Dest_dow": df["Dest"].astype(str) + "_" + df["DayOfWeek"].astype(str),
        "Carrier_dow": df["UniqueCarrier"].astype(str) + "_" + df["DayOfWeek"].astype(str),
        "route_bin5": df["Origin"].astype(str) + "_" + df["Dest"].astype(str) + "_" + bin5,
        "Origin_bin10": df["Origin"].astype(str) + "_" + bin10,
        "Dest_bin10": df["Dest"].astype(str) + "_" + bin10,
        "Carrier_bin10": df["UniqueCarrier"].astype(str) + "_" + bin10,
        "Origin_bin5": df["Origin"].astype(str) + "_" + bin5,
        "Dest_bin5": df["Dest"].astype(str) + "_" + bin5,
        "Carrier_bin5": df["UniqueCarrier"].astype(str) + "_" + bin5,
        "carrier_origin": df["UniqueCarrier"].astype(str) + "_" + df["Origin"].astype(str),
        "carrier_dest": df["UniqueCarrier"].astype(str) + "_" + df["Dest"].astype(str),
        "route_dow": df["Origin"].astype(str) + "_" + df["Dest"].astype(str) + "_" + df["DayOfWeek"].astype(str),
        "carrier_route_bin": df["UniqueCarrier"].astype(str) + "_" + df["Origin"].astype(str) + "_" + df["Dest"].astype(str) + "_" + bin30,
        "carrier_route_bin15": df["UniqueCarrier"].astype(str) + "_" + df["Origin"].astype(str) + "_" + df["Dest"].astype(str) + "_" + bin15,
        "carrier_origin_bin": df["UniqueCarrier"].astype(str) + "_" + df["Origin"].astype(str) + "_" + bin30,
        "carrier_dest_bin": df["UniqueCarrier"].astype(str) + "_" + df["Dest"].astype(str) + "_" + bin30,
        "carrier_route_bin5": df["UniqueCarrier"].astype(str) + "_" + df["Origin"].astype(str) + "_" + df["Dest"].astype(str) + "_" + bin5,
        "carrier_origin_bin15": df["UniqueCarrier"].astype(str) + "_" + df["Origin"].astype(str) + "_" + bin15,
        "carrier_dest_bin15": df["UniqueCarrier"].astype(str) + "_" + df["Dest"].astype(str) + "_" + bin15,
    }


def te_stats(keys: pd.Series, y: np.ndarray) -> pd.Series:
    g = pd.DataFrame({"k": keys.values, "y": y}).groupby("k")["y"].agg(["sum", "count"])
    return (g["sum"] + prior * SMOOTH) / (g["count"] + SMOOTH)


train_keys = make_keys(train)
full_maps = {c: te_stats(train_keys[c], y_train) for c in TE_COLS}
freq_maps = {c: train_keys[c].value_counts() for c in TE_COLS}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering lives here so predict_proba reproduces it on unseen rows.
    X = df[feature_cols].copy()
    # high-cardinality raw categoricals are replaced by target/frequency encodings below
    X = X.drop(columns=cat_cols)
    for c in ["Month", "DayofMonth", "DayOfWeek"]:
        X[c + "_num"] = df[c].astype(str).str.replace("c-", "", regex=False).astype(float)
    keys = make_keys(df)
    for c in TE_COLS:
        X[c + "_te"] = keys[c].map(full_maps[c]).fillna(prior).astype(float)
        X[c + "_freq"] = np.log1p(keys[c].map(freq_maps[c]).fillna(0)).astype(float)
    return X


# --- model: ensemble of diverse shallow XGBoost models ------------------------
VARIANTS = [
    dict(max_depth=3, n_estimators=800, colsample_bytree=0.7, subsample=0.9, min_child_weight=5),
    dict(max_depth=3, n_estimators=800, colsample_bytree=0.8, subsample=0.85, min_child_weight=5),
    dict(max_depth=3, n_estimators=900, colsample_bytree=0.9, subsample=0.9, min_child_weight=5),
    dict(max_depth=3, n_estimators=700, colsample_bytree=1.0, subsample=0.9, min_child_weight=5),
    dict(max_depth=4, n_estimators=300, colsample_bytree=0.8, subsample=0.9, min_child_weight=5),
    dict(max_depth=4, n_estimators=250, colsample_bytree=0.9, subsample=0.85, min_child_weight=5),
    dict(max_depth=3, n_estimators=900, colsample_bytree=0.6, subsample=0.8, min_child_weight=3),
    dict(max_depth=3, n_estimators=1000, colsample_bytree=0.8, subsample=0.9, min_child_weight=10),
    dict(max_depth=4, n_estimators=400, colsample_bytree=0.7, subsample=0.85, min_child_weight=3),
    dict(max_depth=4, n_estimators=500, colsample_bytree=1.0, subsample=0.9, min_child_weight=10),
    dict(max_depth=5, n_estimators=200, colsample_bytree=0.7, subsample=0.85, min_child_weight=10),
    dict(max_depth=5, n_estimators=250, colsample_bytree=0.8, subsample=0.9, min_child_weight=5),
    dict(max_depth=5, n_estimators=150, colsample_bytree=0.9, subsample=0.9, min_child_weight=20),
]
N_MODELS = 13

X_train = prepare(train)
# replace target-encoded columns with out-of-fold values to avoid leakage during training
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
for c in TE_COLS:
    oof = np.full(len(train), prior, dtype=float)
    for tr_idx, va_idx in kf.split(train):
        st = te_stats(train_keys[c].iloc[tr_idx], y_train[tr_idx])
        oof[va_idx] = train_keys[c].iloc[va_idx].map(st).fillna(prior).to_numpy()
    X_train[c + "_te"] = oof

models = []
t0 = time.time()
for i in range(N_MODELS):
    cfg = VARIANTS[i % len(VARIANTS)]
    m = xgb.XGBClassifier(
        learning_rate=0.03,
        reg_lambda=1.0,
        tree_method="hist",
        enable_categorical=True,
        random_state=SEED + i,
        n_jobs=N_JOBS,
        **cfg,
    )
    m.fit(X_train, y_train)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"DEBUG Eval AUC: {eval_auc:.6f}")
print(f"Eval AUC: {eval_auc:.4f}")
