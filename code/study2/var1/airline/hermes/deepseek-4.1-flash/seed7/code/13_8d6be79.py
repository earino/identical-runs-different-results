"""XGBoost binary classifier (airline delay). THIS IS THE ONLY FILE THE AGENT EDITS.

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

# --- feature definitions (fitted on TRAIN only, applied by prepare()) -----------
# The raw calendar columns (Month / DayofMonth / DayOfWeek) are deliberately NOT used: their seasonal and
# day-of-month splits are year-specific and cost ~0.008 AUC on the later-year eval set. The weekend flag is
# dropped too: it is worth ~-0.001 once the clock features are present.
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
# Half the bag drops the high-cardinality Origin / Dest identity columns (their frequency and hub-size stats
# stay): they are the least transferable features we have, so the two sub-bags make different mistakes and
# average better together.
OD_CATS = ["Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}

# Structural stats fitted on train only. Frequency counts and hub-size stats act as stable "busyness"
# proxies for carrier / airport / city-pair and transfer across years far better than identity alone.
FREQ_SRC = ["UniqueCarrier", "Origin", "Dest"]
_route = train["Origin"].astype(str) + "_" + train["Dest"].astype(str)
freq_maps = {c: train[c].value_counts().astype(float) for c in FREQ_SRC}
freq_maps["route"] = _route.value_counts().astype(float)

HUB_STATS = {
    "o_ndest": ("Origin", train.groupby("Origin")["Dest"].nunique().astype(float)),
    "d_norig": ("Dest", train.groupby("Dest")["Origin"].nunique().astype(float)),
    "o_ncarrier": ("Origin", train.groupby("Origin")["UniqueCarrier"].nunique().astype(float)),
    "c_ndest": ("UniqueCarrier", train.groupby("UniqueCarrier")["Dest"].nunique().astype(float)),
}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    X["Distance"] = pd.to_numeric(df["Distance"], errors="coerce").astype(float)
    X["log_dist"] = np.log1p(X["Distance"])
    # scheduled departure hhmm -> clock features (time of day is the strongest single signal)
    t = pd.to_numeric(df["DepTime"], errors="coerce").fillna(0.0)
    hour = (t // 100).clip(0, 23)
    minute = (t % 100).clip(0, 59)
    mod = hour * 60 + minute
    X["dep_hour"] = hour.astype(float)
    X["dep_sin"] = np.sin(2.0 * np.pi * mod / 1440.0)
    X["dep_cos"] = np.cos(2.0 * np.pi * mod / 1440.0)
    # Approximate scheduled ARRIVAL time: departure clock time + flight time estimated from distance
    # (~460 mph cruise plus ~30 min taxi/climb/descent). Late-evening arrivals pick up the day's accumulated
    # delay, and this is the single best feature addition so far (+0.004 AUC).
    duration_h = X["Distance"] / 460.0 + 0.5
    arr_hour = (mod / 60.0 + duration_h) % 24.0
    X["arr_hour"] = arr_hour
    X["arr_sin"] = np.sin(2.0 * np.pi * arr_hour / 24.0)
    X["arr_cos"] = np.cos(2.0 * np.pi * arr_hour / 24.0)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    route = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    for c in FREQ_SRC:
        X[c + "_n"] = df[c].map(freq_maps[c]).fillna(0.0).astype(float)
    X["route_n"] = route.map(freq_maps["route"]).fillna(0.0).astype(float)
    for name, (col, mapping) in HUB_STATS.items():
        X[name] = df[col].map(mapping).fillna(0.0).astype(float)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
# Bag of deep trees with strong column subsampling. Depth is the dominant knob on this dataset (the shallow
# baseline was badly underfit); the column subsample plus seed/depth averaging is what stops the deep trees
# from memorising year-specific noise.
PARAMS = dict(
    n_estimators=55,
    learning_rate=0.05,
    tree_method="hist",
    enable_categorical=True,
    n_jobs=N_JOBS,
)
MODEL_SPECS = [(8, 0.4), (12, 0.4), (16, 0.45), (20, 0.5), (24, 0.5), (28, 0.5), (32, 0.45), (36, 0.4), (40, 0.4)]

t0 = time.time()
models = []      # (model, uses_origin_dest_cols)
for depth, colsample in MODEL_SPECS:
    for i, (drop_od, seed_off) in enumerate([(False, 0), (True, 7)]):
        m = xgb.XGBClassifier(
            max_depth=depth,
            colsample_bytree=colsample,
            random_state=SEED + seed_off,
            **PARAMS,
        )
        Xtr = prepare(train)
        if drop_od:
            Xtr = Xtr.drop(columns=OD_CATS)
        m.fit(Xtr, to_y(train))
        models.append((m, not drop_od))
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    X_no_od = X.drop(columns=OD_CATS)
    return np.mean([m.predict_proba(X if od else X_no_od)[:, 1] for m, od in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
