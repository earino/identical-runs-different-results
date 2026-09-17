"""Airline delay XGBoost classifier (agent-iterated). Contract: see program.md."""
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

# --- fixed statistics learned from TRAIN ONLY ---------------------------------
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}


def _tods(dep: pd.Series) -> pd.Series:
    return ((dep // 100 % 24) * 60 + dep % 100).astype("int64") % 1440


def _route(df: pd.DataFrame) -> pd.Series:
    return df["Origin"].astype(str) + "_" + df["Dest"].astype(str)


# frequency maps from train
tmp_tr = pd.DataFrame({"hour": _tods(train["DepTime"]), "route": _route(train)})
freq_maps = {c: train[c].value_counts() for c in ["UniqueCarrier", "Origin", "Dest"]}
freq_maps["route"] = tmp_tr["route"].value_counts()
freq_maps["hour"] = tmp_tr["hour"].value_counts()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """ALL feature engineering lives here; uses only fixed train-fitted maps."""
    X = pd.DataFrame(index=df.index)
    mon = df["Month"].str[2:].astype("int64")
    dom = df["DayofMonth"].str[2:].astype("int64")
    dow = df["DayOfWeek"].str[2:].astype("int64")
    tod = _tods(df["DepTime"])
    X["month"] = mon
    X["dom"] = dom
    X["dow"] = dow
    X["hour"] = tod // 60
    X["minute"] = tod % 60
    ang = 2 * np.pi * tod / 1440.0
    X["tod_sin"] = np.sin(ang)
    X["tod_cos"] = np.cos(ang)
    X["month_sin"] = np.sin(2 * np.pi * mon / 12)
    X["month_cos"] = np.cos(2 * np.pi * mon / 12)
    X["doy"] = (pd.to_datetime(dict(year=2001, month=mon, day=dom), errors="coerce")
                .map(lambda d: d.timetuple().tm_yday if pd.notna(d) else np.nan))
    X["doy"] = X["doy"].fillna(mon * 30 + dom).astype("int64")
    X["weekend"] = (dow >= 6).astype("int64")
    X["distance"] = df["Distance"]
    X["distance_log"] = np.log1p(df["Distance"])
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    route = _route(df)
    X["route"] = pd.Categorical(route, categories=sorted(tmp_tr["route"].unique()))
    X["freq_carrier"] = df["UniqueCarrier"].map(freq_maps["UniqueCarrier"]).fillna(0).astype("float64")
    X["freq_origin"] = df["Origin"].map(freq_maps["Origin"]).fillna(0).astype("float64")
    X["freq_dest"] = df["Dest"].map(freq_maps["Dest"]).fillna(0).astype("float64")
    X["freq_route"] = route.map(freq_maps["route"]).fillna(0).astype("float64")
    X["freq_hour"] = tod.map(freq_maps["hour"]).fillna(0).astype("float64")
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
y = to_y(train)
X_tr = prepare(train)
X_ev = prepare(evald)

model = xgb.XGBClassifier(
    n_estimators=30,
    max_depth=6,
    learning_rate=0.1,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(X_tr, y)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
