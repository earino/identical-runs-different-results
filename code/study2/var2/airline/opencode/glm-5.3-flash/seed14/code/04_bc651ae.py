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
CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
TE_COLS = ["UniqueCarrier", "Origin", "Dest", "route"]
M_SMOOTH = 30.0
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}


def _route(df: pd.DataFrame) -> pd.Series:
    return df["Origin"].astype(str) + "_" + df["Dest"].astype(str)


def _tods(dep: pd.Series) -> pd.Series:
    return ((dep // 100 % 24) * 60 + dep % 100).astype("int64") % 1440


y_tr = (train[TARGET] == POSITIVE).astype(int).to_numpy()
prior = float(y_tr.mean())
route_tr = _route(train)
te_maps = {}
for c, series in [("UniqueCarrier", train["UniqueCarrier"]), ("Origin", train["Origin"]),
                  ("Dest", train["Dest"]), ("route", route_tr)]:
    g = pd.Series(y_tr).groupby(series.to_numpy()).agg(["sum", "count"])
    te_maps[c] = ((g["sum"] + M_SMOOTH * prior) / (g["count"] + M_SMOOTH)).to_dict()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """ALL feature engineering lives here; uses only fixed train-fitted maps."""
    X = pd.DataFrame(index=df.index)
    X["DepTime"] = df["DepTime"]
    X["Distance"] = df["Distance"]
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    route = _route(df)
    for c, series in [("UniqueCarrier", df["UniqueCarrier"]), ("Origin", df["Origin"]),
                      ("Dest", df["Dest"]), ("route", route)]:
        X[f"te_{c}"] = series.map(te_maps[c]).fillna(prior).astype("float64")
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
y = to_y(train)
X_tr = prepare(train)
X_ev = prepare(evald)

# out-of-fold target encoding for the training matrix (leak-free)
rng = np.random.RandomState(SEED)
kfold = rng.permutation(np.arange(len(y)) % 5)
for c, series in [("UniqueCarrier", train["UniqueCarrier"]), ("Origin", train["Origin"]),
                  ("Dest", train["Dest"]), ("route", route_tr)]:
    col = series.to_numpy()
    oof = np.empty(len(y))
    for k in range(5):
        msk = kfold == k
        g = pd.Series(y[~msk]).groupby(col[~msk]).agg(["sum", "count"])
        enc = (g["sum"] + M_SMOOTH * prior) / (g["count"] + M_SMOOTH)
        oof[msk] = pd.Series(col[msk]).map(enc).fillna(prior).to_numpy()
    X_tr[f"te_{c}"] = oof

model = xgb.XGBClassifier(
    n_estimators=20,
    max_depth=5,
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
