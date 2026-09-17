"""XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS."""
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

# --- feature definitions ------------------------------------------------------
CATS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
# categorical levels from TRAIN ONLY; unseen levels -> NaN inside prepare()
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CATS}
cat_levels["TODbin"] = pd.Index(range(48))

# stable count features fitted on TRAIN only
_route_tr = train["Origin"].astype(str) + "_" + train["Dest"].astype(str)
FREQ = {
    "Origin": train["Origin"].value_counts(),
    "Dest": train["Dest"].value_counts(),
    "Carrier": train["UniqueCarrier"].value_counts(),
    "Route": _route_tr.value_counts(),
}
_h = (train["DepTime"].astype("int64") // 100) % 24
_tod = _h * 60 + (train["DepTime"].astype("int64") % 100).clip(upper=59)
_tb = (_tod // 30).astype(str)
FREQ["OC"] = (train["Origin"] + "_" + train["UniqueCarrier"]).value_counts()
FREQ["CT"] = (train["UniqueCarrier"] + "_" + _tb).value_counts()
FREQ["OT"] = (train["Origin"] + "_" + _tb).value_counts()
FREQ["DT"] = (train["Dest"] + "_" + _tb).value_counts()
FREQ["RT"] = (_route_tr + "_" + _tb).value_counts()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering lives here so predict_proba reproduces it on unseen rows.
    X = pd.DataFrame(index=df.index)
    for c in CATS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    h = (df["DepTime"].astype("int64") // 100) % 24
    m = (df["DepTime"].astype("int64") % 100).clip(upper=59)
    tod = h * 60 + m
    route = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    X["TOD"] = tod.astype("int32")
    X["H"] = h.astype("int16")
    X["TODbin"] = pd.Categorical(tod // 30, categories=cat_levels["TODbin"])
    X["TOD_sin"] = np.sin(2 * np.pi * tod / 1440.0)
    X["TOD_cos"] = np.cos(2 * np.pi * tod / 1440.0)
    X["Dist"] = df["Distance"].astype("float32")
    X["LogD"] = np.log1p(df["Distance"].astype("float32"))
    X["freq_Origin"] = df["Origin"].map(FREQ["Origin"]).fillna(0).astype("float32")
    X["freq_Dest"] = df["Dest"].map(FREQ["Dest"]).fillna(0).astype("float32")
    X["freq_Carrier"] = df["UniqueCarrier"].map(FREQ["Carrier"]).fillna(0).astype("float32")
    X["freq_Route"] = route.map(FREQ["Route"]).fillna(0).astype("float32")
    tb = (tod // 30).astype(str)
    X["freq_OC"] = (df["Origin"] + "_" + df["UniqueCarrier"]).map(FREQ["OC"]).fillna(0).astype("float32")
    X["freq_CT"] = (df["UniqueCarrier"] + "_" + tb).map(FREQ["CT"]).fillna(0).astype("float32")
    X["freq_OT"] = (df["Origin"] + "_" + tb).map(FREQ["OT"]).fillna(0).astype("float32")
    X["freq_DT"] = (df["Dest"] + "_" + tb).map(FREQ["DT"]).fillna(0).astype("float32")
    X["freq_RT"] = (route + "_" + tb).map(FREQ["RT"]).fillna(0).astype("float32")
    fo = X["freq_Origin"] + 1.0
    X["share_Route_O"] = X["freq_Route"] / fo
    X["share_OC_O"] = X["freq_OC"] / fo
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model ensemble -----------------------------------------------------------
COMMON = dict(
    n_estimators=600,
    max_depth=30,
    max_leaves=1023,
    learning_rate=0.03,
    min_child_weight=1,
    subsample=1.0,
    colsample_bytree=0.3,
    reg_lambda=5.0,
    reg_alpha=0.0,
    gamma=0.5,
    max_bin=256,
    max_cat_to_onehot=1,
    grow_policy="lossguide",
    tree_method="hist",
    enable_categorical=True,
    n_jobs=N_JOBS,
    early_stopping_rounds=25,
    eval_metric="auc",
)
SEEDS = [42, 1]
models = []
BEST_ITS = []
Xtr, ytr = prepare(train), to_y(train)
Xev, yev = prepare(evald), to_y(evald)
t0 = time.time()
for s in SEEDS:
    m = xgb.XGBClassifier(random_state=s, **COMMON)
    m.fit(Xtr, ytr, eval_set=[(Xev, yev)], verbose=False)
    models.append(m)
    BEST_ITS.append(m.best_iteration + 1)
    print(f"  model seed={s} best_iter={m.best_iteration} best_auc={m.best_score:.4f}")
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    ps = [m.predict_proba(X, iteration_range=(0, bi))[:, 1] for m, bi in zip(models, BEST_ITS)]
    return np.mean(ps, axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
