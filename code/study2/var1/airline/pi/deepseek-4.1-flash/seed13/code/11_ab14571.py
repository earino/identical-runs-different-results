"""XGBoost binary classifier for airline delay prediction. ONLY FILE THE AGENT EDITS."""
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

RAW_CAT = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]

cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in RAW_CAT}
cat_levels["HourCat"] = pd.Index(range(24))


def _ci(s: pd.Series) -> np.ndarray:
    """c-<n> string -> integer."""
    return pd.to_numeric(s.astype(str).str.replace("c-", "", regex=False), errors="coerce").fillna(1).astype(int).to_numpy()


def _hour(df: pd.DataFrame) -> np.ndarray:
    dep = pd.to_numeric(df["DepTime"], errors="coerce").fillna(0).to_numpy()
    return np.clip(dep // 100, 0, 23)


def _freq_counts(df: pd.DataFrame) -> dict:
    route = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    hour_s = pd.Series(_hour(df), index=df.index).astype(str)
    return {
        "F_UniqueCarrier": df["UniqueCarrier"].value_counts(),
        "F_Origin": df["Origin"].value_counts(),
        "F_Dest": df["Dest"].value_counts(),
        "F_Route": route.value_counts(),
        "F_Hour": pd.Series(_hour(df)).value_counts(),
        "F_Org_Hour": (df["Origin"].astype(str) + "_" + hour_s).value_counts(),
        "F_UC_Hour": (df["UniqueCarrier"].astype(str) + "_" + hour_s).value_counts(),
        "F_Dst_Hour": (df["Dest"].astype(str) + "_" + hour_s).value_counts(),
        "F_Route_Hour": (route + "_" + hour_s).value_counts(),
    }


FREQ = _freq_counts(train)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    for c in RAW_CAT:
        X[c] = pd.Categorical(df[c].astype(str), categories=cat_levels[c])
    X["HourCat"] = pd.Categorical(_hour(df), categories=cat_levels["HourCat"])

    dep = pd.to_numeric(df["DepTime"], errors="coerce").fillna(0).to_numpy()
    hour = _hour(df).astype(float)
    minute = np.where(dep % 100 < 60, dep % 100, 0).astype(float)
    month = _ci(df["Month"]).astype(float)
    day = _ci(df["DayofMonth"]).astype(float)
    dow = _ci(df["DayOfWeek"]).astype(float)
    doy = pd.to_datetime(
        "2005-" + np.char.zfill(month.astype(int).astype(str), 2) + "-"
        + np.char.zfill(day.astype(int).astype(str), 2),
        errors="coerce",
    ).dayofyear.to_numpy(dtype=float)

    X["Hour"] = hour
    X["Minute"] = minute
    X["DepTimeNum"] = hour * 60 + minute
    X["RoundDep"] = hour * 4 + np.round(minute / 15)
    X["MonthNum"] = month
    X["DayNum"] = day
    X["DowNum"] = dow
    X["DayOfYear"] = doy
    X["IsWeekend"] = (dow >= 6).astype(float)
    X["HourSin"] = np.sin(2 * np.pi * hour / 24)
    X["HourCos"] = np.cos(2 * np.pi * hour / 24)
    dist = pd.to_numeric(df["Distance"], errors="coerce").fillna(0).to_numpy(dtype=float)
    X["Distance"] = dist
    X["LogDistance"] = np.log1p(dist)

    route = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    hour_s = pd.Series(_hour(df), index=df.index).astype(str)
    X["F_UniqueCarrier"] = df["UniqueCarrier"].map(FREQ["F_UniqueCarrier"]).fillna(0).to_numpy(dtype=float)
    X["F_Origin"] = df["Origin"].map(FREQ["F_Origin"]).fillna(0).to_numpy(dtype=float)
    X["F_Dest"] = df["Dest"].map(FREQ["F_Dest"]).fillna(0).to_numpy(dtype=float)
    X["F_Route"] = route.map(FREQ["F_Route"]).fillna(0).to_numpy(dtype=float)
    X["F_Hour"] = pd.Series(_hour(df)).map(FREQ["F_Hour"]).fillna(0).to_numpy(dtype=float)
    X["F_Org_Hour"] = (df["Origin"].astype(str) + "_" + hour_s).map(FREQ["F_Org_Hour"]).fillna(0).to_numpy(dtype=float)
    X["F_UC_Hour"] = (df["UniqueCarrier"].astype(str) + "_" + hour_s).map(FREQ["F_UC_Hour"]).fillna(0).to_numpy(dtype=float)
    X["F_Dst_Hour"] = (df["Dest"].astype(str) + "_" + hour_s).map(FREQ["F_Dst_Hour"]).fillna(0).to_numpy(dtype=float)
    X["F_Route_Hour"] = (route + "_" + hour_s).map(FREQ["F_Route_Hour"]).fillna(0).to_numpy(dtype=float)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# Diverse deep-tree ensemble; deep trees + low colsample captured strong interactions.
# Lower learning rate with more trees gave the best single-model AUC on the held-out year.
MODEL_PARAMS = [
    dict(max_depth=12, learning_rate=0.06, n_estimators=200),
    dict(max_depth=18, learning_rate=0.05, n_estimators=200, min_child_weight=0),
    dict(max_depth=22, learning_rate=0.05, n_estimators=200, min_child_weight=0),
    dict(max_depth=28, learning_rate=0.05, n_estimators=200),
]

X_train = prepare(train)
y_train = to_y(train)
BASE_PARAMS = dict(
    tree_method="hist",
    enable_categorical=True,
    subsample=1.0,
    min_child_weight=1,
    reg_lambda=1.0,
    colsample_bytree=0.4,
    max_cat_to_onehot=64,
    n_jobs=N_JOBS,
)
models = []
t0 = time.time()
for i, params in enumerate(MODEL_PARAMS):
    m = xgb.XGBClassifier(random_state=SEED + i, **{**BASE_PARAMS, **params})
    m.fit(X_train, y_train)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
