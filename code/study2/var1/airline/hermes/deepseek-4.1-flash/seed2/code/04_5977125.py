"""DIAGNOSTIC SWEEP (temporary): add-on features on top of the current best model.
Measures Eval AUC for frequency/traffic-count and interaction features."""
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
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")
y_tr = (train[TARGET] == POSITIVE).astype(int).to_numpy()
y_ev = (evald[TARGET] == POSITIVE).astype(int).to_numpy()

C_NUM_COLS = ["Month", "DayofMonth", "DayOfWeek"]
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}


def base(df):
    X = pd.DataFrame(index=df.index)
    for c in C_NUM_COLS:
        X[c] = df[c].str.replace("c-", "", regex=False).astype(np.int16)
    X["Distance"] = df["Distance"].astype(np.float32)
    dep = df["DepTime"].astype(np.int32)
    hour = (dep // 100).clip(0, 23)
    minute = (dep % 100).clip(0, 59)
    X["DepHour"] = hour.astype(np.int16)
    X["DepMinOfDay"] = (hour * 60 + minute).astype(np.int16)
    X["HourSin"] = np.sin(2 * np.pi * X["DepHour"] / 24).astype(np.float32)
    X["HourCos"] = np.cos(2 * np.pi * X["DepHour"] / 24).astype(np.float32)
    doy = (X["Month"] - 1) * 30 + X["DayofMonth"]
    X["DoySin"] = np.sin(2 * np.pi * doy / 365).astype(np.float32)
    X["DoyCos"] = np.cos(2 * np.pi * doy / 365).astype(np.float32)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    return X


def _keys(df):
    dep = df["DepTime"].astype(np.int32)
    h = (dep // 100).clip(0, 23)
    return {
        "carrier": df["UniqueCarrier"],
        "origin": df["Origin"],
        "dest": df["Dest"],
        "route": df["Origin"] + "_" + df["Dest"],
        "orig_hour": df["Origin"] + "_" + h.astype(str),
        "dest_hour": df["Dest"] + "_" + h.astype(str),
        "carrier_hour": df["UniqueCarrier"] + "_" + h.astype(str),
    }


KEY_TR = _keys(train)
KEY_EV = _keys(evald)
CNT = {k: KEY_TR[k].value_counts() for k in KEY_TR}
cnt_lv = {k: CNT[k].index for k in CNT}


def add_counts(X, df, keys, names):
    K = _keys(df)
    for k in names:
        v = K[k].map(CNT[k])
        X["cnt_" + k] = np.log1p(v.fillna(0).to_numpy()).astype(np.float32)
    return X


def add_offline_cats(X, df):
    dep = df["DepTime"].astype(np.int32)
    h = (dep // 100).clip(0, 23)
    lv_ch = pd.Index(sorted((train["UniqueCarrier"] + "_" + (train["DepTime"] // 100).clip(0, 23).astype(str)).unique()))
    X["CarrierHour"] = pd.Categorical(df["UniqueCarrier"] + "_" + h.astype(str), categories=lv_ch)
    return X


def v0(df):
    return base(df)


def v1(df):
    return add_counts(base(df), df, None, ["carrier", "origin", "dest", "route"])


def v2(df):
    return add_counts(base(df), df, None, ["orig_hour", "dest_hour", "carrier_hour"])


def v3(df):
    return add_counts(base(df), df, None, ["carrier", "origin", "dest", "route", "orig_hour", "dest_hour", "carrier_hour"])


def v4(df):
    return add_offline_cats(v3(df), df)


def v5(df):
    return add_offline_cats(base(df), df)


VARIANTS = {"V0_base": v0, "V1_counts": v1, "V2_hourcounts": v2, "V3_allcounts": v3,
            "V4_allcounts+ch": v4, "V5_carrierhour": v5}

MODEL = dict(n_estimators=800, max_depth=5, learning_rate=0.03, min_child_weight=30,
             subsample=0.7, colsample_bytree=0.5, reg_lambda=5.0)

res = {}
for name, fn in VARIANTS.items():
    Xtr, Xev = fn(train), fn(evald)
    m = xgb.XGBClassifier(tree_method="hist", enable_categorical=True, random_state=SEED,
                          n_jobs=N_JOBS, **MODEL)
    t0 = time.time()
    m.fit(Xtr, y_tr, verbose=False)
    auc = roc_auc_score(y_ev, m.predict_proba(Xev)[:, 1])
    res[name] = auc
    print(f"RESULT {name:18s} ncol={Xtr.shape[1]:3d} auc={auc:.4f} t={time.time()-t0:.1f}s", flush=True)

print("\n=== ranked ===")
for k, v in sorted(res.items(), key=lambda kv: -kv[1]):
    print(f"{v:.4f}  {k}")

best = max(res.items(), key=lambda kv: kv[1])
print(f"Eval AUC: {best[1]:.4f}")


def predict_proba(df):
    return np.zeros(len(df))
