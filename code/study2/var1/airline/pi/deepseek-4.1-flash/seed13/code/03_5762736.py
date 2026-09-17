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
NUM_COLS = [
    "Hour", "Minute", "DepTimeNum", "MonthNum", "DayNum", "DowNum",
    "DayOfYear", "IsWeekend", "HourSin", "HourCos", "Distance", "LogDistance",
    "F_UniqueCarrier", "F_Origin", "F_Dest", "F_Route", "F_Hour", "F_Org_Hour",
]

cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in RAW_CAT}


def _ci(s: pd.Series) -> np.ndarray:
    """c-<n> string -> integer."""
    return pd.to_numeric(s.astype(str).str.replace("c-", "", regex=False), errors="coerce").fillna(1).astype(int).to_numpy()


def _hour(df: pd.DataFrame) -> np.ndarray:
    dep = pd.to_numeric(df["DepTime"], errors="coerce").fillna(0).to_numpy()
    return np.clip(dep // 100, 0, 23)


def _freq_counts(df: pd.DataFrame) -> dict:
    route = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    hour = _hour(df)
    org_hour = df["Origin"].astype(str) + "_" + pd.Series(hour, index=df.index).astype(str)
    return {
        "F_UniqueCarrier": df["UniqueCarrier"].value_counts(),
        "F_Origin": df["Origin"].value_counts(),
        "F_Dest": df["Dest"].value_counts(),
        "F_Route": route.value_counts(),
        "F_Hour": pd.Series(hour).value_counts(),
        "F_Org_Hour": org_hour.value_counts(),
    }


FREQ = _freq_counts(train)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    for c in RAW_CAT:
        X[c] = pd.Categorical(df[c].astype(str), categories=cat_levels[c])

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
    X["F_UniqueCarrier"] = df["UniqueCarrier"].map(FREQ["F_UniqueCarrier"]).fillna(0).to_numpy(dtype=float)
    X["F_Origin"] = df["Origin"].map(FREQ["F_Origin"]).fillna(0).to_numpy(dtype=float)
    X["F_Dest"] = df["Dest"].map(FREQ["F_Dest"]).fillna(0).to_numpy(dtype=float)
    X["F_Route"] = route.map(FREQ["F_Route"]).fillna(0).to_numpy(dtype=float)
    X["F_Hour"] = pd.Series(_hour(df)).map(FREQ["F_Hour"]).fillna(0).to_numpy(dtype=float)
    org_hour = df["Origin"].astype(str) + "_" + pd.Series(_hour(df), index=df.index).astype(str)
    X["F_Org_Hour"] = org_hour.map(FREQ["F_Org_Hour"]).fillna(0).to_numpy(dtype=float)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


model = xgb.XGBClassifier(
    n_estimators=250,
    max_depth=5,
    learning_rate=0.05,
    subsample=1.0,
    colsample_bytree=0.4,
    min_child_weight=5,
    reg_lambda=1.0,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(prepare(train), to_y(train))
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
