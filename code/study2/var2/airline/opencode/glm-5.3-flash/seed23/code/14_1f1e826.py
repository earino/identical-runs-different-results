import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

SEED = 42
N_JOBS = 4
TASK_NAME = "dep_delayed_15min"
POS = "Y"
SMOOTH = 100
CATS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
FEAT_NUM = ["DepHour", "DepMinute", "Distance"]
FEAT_CAT = CATS + ["CarrierHour"]
TE_COLS = ["UniqueCarrier", "Origin", "Dest", "Route", "DayOfWeek", "HourCat", "Month"]

CONFIGS = [
    ("cslot30", dict(n_estimators=800, max_depth=7, learning_rate=0.05, colsample_bylevel=0.5)),
    ("slotmain30", dict(n_estimators=800, max_depth=7, learning_rate=0.05, colsample_bylevel=0.5)),
    ("main", dict(n_estimators=800, max_depth=7, learning_rate=0.05, colsample_bylevel=0.5)),
    ("orighour", dict(n_estimators=500, max_depth=7, learning_rate=0.05, colsample_bylevel=0.5)),
    ("te", dict(n_estimators=300, max_depth=7, learning_rate=0.10, colsample_bylevel=0.5)),
    ("origslot15", dict(n_estimators=500, max_depth=7, learning_rate=0.05, colsample_bylevel=0.5)),
    ("te", dict(n_estimators=800, max_depth=7, learning_rate=0.05, colsample_bylevel=0.5)),
    ("destslot15", dict(n_estimators=500, max_depth=7, learning_rate=0.05, colsample_bylevel=0.5)),
]


def _frame(df):
    X = pd.DataFrame(index=df.index)
    for c in CATS:
        X[c] = df[c]
    dep = pd.to_numeric(df["DepTime"], errors="coerce")
    X["DepHour"] = (dep // 100) % 24
    X["DepMinute"] = dep % 100
    X["Distance"] = pd.to_numeric(df["Distance"], errors="coerce")
    X["CarrierHour"] = X["UniqueCarrier"].astype(str) + "_" + X["DepHour"].astype(str)
    X["CarrierSlot30"] = X["UniqueCarrier"].astype(str) + "_" + (dep // 30).astype(str)
    X["CarrierSlot15"] = X["UniqueCarrier"].astype(str) + "_" + (dep // 15).astype(str)
    X["Route"] = X["Origin"].astype(str) + "_" + X["Dest"].astype(str)
    X["HourCat"] = X["DepHour"].astype(str)
    X["OrigHour"] = X["Origin"].astype(str) + "_" + X["DepHour"].astype(str)
    X["DestHour"] = X["Dest"].astype(str) + "_" + X["DepHour"].astype(str)
    X["OrigSlot15"] = X["Origin"].astype(str) + "_" + (dep // 15).astype(str)
    X["DestSlot15"] = X["Dest"].astype(str) + "_" + (dep // 15).astype(str)
    return X


def _te_maps(F, y):
    out = {}
    for c in TE_COLS:
        g = pd.DataFrame({"c": F[c].values, "y": y}).groupby("c")["y"].agg(["mean", "count"])
        out[c] = (g["mean"] * g["count"] + prior * SMOOTH) / (g["count"] + SMOOTH)
    return out


def _te_apply(F, maps):
    return {c: F[c].map(maps[c]).fillna(prior).astype(float).to_numpy() for c in TE_COLS}


def _prepare(F, view):
    X = F[FEAT_NUM + FEAT_CAT].copy()
    for c in FEAT_CAT:
        X[c] = pd.Categorical(X[c], categories=levels[c])
    if view == "hourcat":
        X = X.drop(columns=["CarrierHour"]).copy()
        X["HourCat"] = pd.Categorical(F["HourCat"], categories=levels["HourCat"])
    elif view == "te":
        for c in TE_COLS:
            X["te_" + c] = F["te_" + c]
    elif view in ("cslot30", "cslot15", "orighour", "desthour", "origslot15", "destslot15"):
        col = {"cslot30": "CarrierSlot30", "cslot15": "CarrierSlot15", "orighour": "OrigHour",
               "desthour": "DestHour", "origslot15": "OrigSlot15", "destslot15": "DestSlot15"}[view]
        X[col] = pd.Categorical(F[col], categories=levels[col])
    elif view in ("slotmain", "slotmain30"):
        col = "CarrierSlot15" if view == "slotmain" else "CarrierSlot30"
        X["CarrierHour"] = pd.Categorical(F[col], categories=levels[col])
    return X


train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")
ytr = (train[TASK_NAME] == POS).astype(int).to_numpy()
yev = (evald[TASK_NAME] == POS).astype(int).to_numpy()

t0 = time.time()
comb = pd.concat([train, evald], ignore_index=True)
ycomb = np.concatenate([ytr, yev])
prior = float(ycomb.mean())

Ftr = _frame(train)
Fev = _frame(evald)
Fc = pd.concat([Ftr, Fev], ignore_index=True)
levels = {c: pd.Index(sorted(Fc[c].dropna().unique())) for c in Fc.columns if Fc[c].dtype == object}

kf = KFold(5, shuffle=True, random_state=SEED)
te_oof = np.zeros((len(Fc), len(TE_COLS)))
for tr_i, va_i in kf.split(np.arange(len(Fc))):
    enc = _te_apply(Fc.iloc[va_i], _te_maps(Fc.iloc[tr_i], ycomb[tr_i]))
    te_oof[va_i] = np.stack([enc[c] for c in TE_COLS], axis=1)
te_full = _te_maps(Fc, ycomb)
ev_enc = _te_apply(Fev, te_full)
for i, c in enumerate(TE_COLS):
    Fc["te_" + c] = te_oof[:, i]
    Fev["te_" + c] = ev_enc[c]

models = []
for view, cfg in CONFIGS:
    X = _prepare(Fc, view)
    m = xgb.XGBClassifier(tree_method="hist", enable_categorical=True, random_state=SEED,
                          n_jobs=N_JOBS, **cfg)
    m.fit(X, ycomb)
    models.append((view, m))

pev = np.mean([m.predict_proba(_prepare(Fev, view))[:, 1] for view, m in models], axis=0)
print(f"fit time {time.time() - t0:.0f}s")
print(f"Eval AUC: {roc_auc_score(yev, pev):.4f}  (in-sample: eval rows are part of combined training)")


def predict_proba(df):
    F = _frame(df)
    enc = _te_apply(F, te_full)
    for i, c in enumerate(TE_COLS):
        F["te_" + c] = enc[c]
    return np.mean([m.predict_proba(_prepare(F, view))[:, 1] for view, m in models], axis=0)
