"""XGBoost binary classifier: predict dep_delayed_15min (Y/N) on airline data.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Layout: 50-model ensemble = 10 configs x 5 feature views.
Views: raw columns; +train-fitted group-mean departure-time deviations;
+day-of-year cyclic; +both; +finer carrier/origin x hour deviation norms.
The 7 strongest configs per view carry a monotone constraint on the first
column (Month), which acts as a strong calendar regularizer under the
2005->2006 shift.
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

BASE_COLS = ["Month", "DayofMonth", "DayOfWeek", "DepTime", "UniqueCarrier", "Origin", "Dest", "Distance"]
CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]

# category levels and group means fitted on TRAIN only
CAT_LEVELS = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}
GRP_MAPS = {
    "dep_m_carrier": train.groupby("UniqueCarrier").DepTime.mean(),
    "dep_m_origin": train.groupby("Origin").DepTime.mean(),
    "dep_m_dest": train.groupby("Dest").DepTime.mean(),
    "dep_m_route": train.assign(_rt=train.Origin.str.cat(train.Dest, sep="_"))
                        .groupby("_rt").DepTime.mean(),
}
_tr_h = train.assign(_hr=np.floor(train.DepTime / 100))
CHR_CARRIER = _tr_h.groupby(["UniqueCarrier", "_hr"]).DepTime.mean()
CHR_ORIGIN = _tr_h.groupby(["Origin", "_hr"]).DepTime.mean()
_tr_rh = train.assign(_rt=train.Origin.str.cat(train.Dest, sep="_"),
                      _hr=np.floor(train.DepTime / 100))
CHR_ROUTE = _tr_rh.groupby(["_rt", "_hr"]).DepTime.mean()
CHR_DEST = _tr_h.groupby(["Dest", "_hr"]).DepTime.mean()
# route x hour delay-rate (target encoding, train-fitted, smoothed)
_tr_rhy = train.assign(_rt=train.Origin.str.cat(train.Dest, sep="_"),
                       _hr=np.floor(train.DepTime / 100),
                       _y=(train[TARGET] == POSITIVE).astype(int))
RT_HR_RATE = _tr_rhy.groupby(["_rt", "_hr"])._y.agg(["sum", "count"])
PRIOR = float((train[TARGET] == POSITIVE).mean())
TE_SMOOTH = 30.0
# route x quarter-day schedule norm
_tr_rq = train.assign(_rt=train.Origin.str.cat(train.Dest, sep="_"),
                      _q=((np.floor(train.DepTime / 100) * 60
                           + (train.DepTime - 100 * np.floor(train.DepTime / 100))) // 180).astype(int))
CHR_ROUTE_Q = _tr_rq.groupby(["_rt", "_q"]).DepTime.mean()


def prepare(df: pd.DataFrame, grp: bool = False, doy: bool = False, chr_: bool = False, rhr: bool = False,
            chrhr: bool = False, triple: bool = False, rte: bool = False, quint: bool = False,
            rhq: bool = False) -> pd.DataFrame:
    """ALL feature engineering lives here; predict_proba calls it on unseen rows.

    Pure transform: every statistic used (category levels, group means) was
    fitted on the training set only.
    """
    X = df[BASE_COLS].copy()
    if grp:
        X["dep_m_carrier"] = df.DepTime - df.UniqueCarrier.map(GRP_MAPS["dep_m_carrier"])
        X["dep_m_origin"] = df.DepTime - df.Origin.map(GRP_MAPS["dep_m_origin"])
        X["dep_m_dest"] = df.DepTime - df.Dest.map(GRP_MAPS["dep_m_dest"])
        X["dep_m_route"] = df.DepTime - (df.Origin.str.cat(df.Dest, sep="_")).map(GRP_MAPS["dep_m_route"])
    if doy:
        mon = df.Month.str.slice(2).astype(int).to_numpy(dtype=np.float64)
        dom = df.DayofMonth.str.slice(2).astype(int).to_numpy(dtype=np.float64)
        doyv = (mon - 1) * 30.44 + dom
        X["doy_sin"] = np.sin(2 * np.pi * doyv / 365.25)
        X["doy_cos"] = np.cos(2 * np.pi * doyv / 365.25)
    if chr_:
        h = np.floor(df.DepTime.to_numpy(dtype=np.float64) / 100)
        X["dep_m_chr"] = df.DepTime - df.assign(_hr=h).set_index(["UniqueCarrier", "_hr"]).index.map(CHR_CARRIER).to_numpy()
        X["dep_m_orh"] = df.DepTime - df.assign(_hr=h).set_index(["Origin", "_hr"]).index.map(CHR_ORIGIN).to_numpy()
    if rhr:
        h = np.floor(df.DepTime.to_numpy(dtype=np.float64) / 100)
        X["dep_m_rhr"] = df.DepTime - df.assign(_rt=df.Origin.str.cat(df.Dest, sep="_"), _hr=h) \
            .set_index(["_rt", "_hr"]).index.map(CHR_ROUTE).to_numpy()
    if chrhr:
        h = np.floor(df.DepTime.to_numpy(dtype=np.float64) / 100)
        rt = df.Origin.str.cat(df.Dest, sep="_")
        X["dep_m_chr"] = df.DepTime - df.assign(_hr=h).set_index(["UniqueCarrier", "_hr"]).index.map(CHR_CARRIER).to_numpy()
        X["dep_m_orh"] = df.DepTime - df.assign(_hr=h).set_index(["Origin", "_hr"]).index.map(CHR_ORIGIN).to_numpy()
        X["dep_m_rhr"] = df.DepTime - df.assign(_rt=rt, _hr=h).set_index(["_rt", "_hr"]).index.map(CHR_ROUTE).to_numpy()
    if triple:
        h = np.floor(df.DepTime.to_numpy(dtype=np.float64) / 100)
        rt = df.Origin.str.cat(df.Dest, sep="_")
        X["dep_m_chr"] = df.DepTime - df.assign(_hr=h).set_index(["UniqueCarrier", "_hr"]).index.map(CHR_CARRIER).to_numpy()
        X["dep_m_orh"] = df.DepTime - df.assign(_hr=h).set_index(["Origin", "_hr"]).index.map(CHR_ORIGIN).to_numpy()
        X["dep_m_dh"] = df.DepTime - df.assign(_hr=h).set_index(["Dest", "_hr"]).index.map(CHR_DEST).to_numpy()
        X["dep_m_rhr"] = df.DepTime - df.assign(_rt=rt, _hr=h).set_index(["_rt", "_hr"]).index.map(CHR_ROUTE).to_numpy()
    if rte or quint:
        h = np.floor(df.DepTime.to_numpy(dtype=np.float64) / 100)
        key = df.assign(_rt=df.Origin.str.cat(df.Dest, sep="_"), _hr=h).set_index(["_rt", "_hr"]).index
        s = key.map(RT_HR_RATE["sum"]).to_numpy(dtype=np.float64)
        c = key.map(RT_HR_RATE["count"]).to_numpy(dtype=np.float64)
        X["rt_hr_rate"] = (s + TE_SMOOTH * PRIOR) / (c + TE_SMOOTH)
    if quint:
        h = np.floor(df.DepTime.to_numpy(dtype=np.float64) / 100)
        rt = df.Origin.str.cat(df.Dest, sep="_")
        X["dep_m_chr"] = df.DepTime - df.assign(_hr=h).set_index(["UniqueCarrier", "_hr"]).index.map(CHR_CARRIER).to_numpy()
        X["dep_m_orh"] = df.DepTime - df.assign(_hr=h).set_index(["Origin", "_hr"]).index.map(CHR_ORIGIN).to_numpy()
        X["dep_m_dh"] = df.DepTime - df.assign(_hr=h).set_index(["Dest", "_hr"]).index.map(CHR_DEST).to_numpy()
        X["dep_m_rhr"] = df.DepTime - df.assign(_rt=rt, _hr=h).set_index(["_rt", "_hr"]).index.map(CHR_ROUTE).to_numpy()
    if rhq:
        rt = df.Origin.str.cat(df.Dest, sep="_")
        q = ((np.floor(df.DepTime / 100) * 60
              + (df.DepTime - 100 * np.floor(df.DepTime / 100))) // 180).astype(int)
        X["dep_m_rhq"] = df.DepTime - df.assign(_rt=rt, _q=q).set_index(["_rt", "_q"]).index.map(CHR_ROUTE_Q).to_numpy()
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=CAT_LEVELS[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


PARAMS = dict(
    n_estimators=120,
    max_depth=4,
    learning_rate=0.05,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

# member configs; the first 7 also get the monotone constraint on column 0
CONFIGS = [
    dict(),
    dict(max_depth=3),
    dict(max_depth=5, n_estimators=60, learning_rate=0.1),
    dict(colsample_bynode=0.7, random_state=1),
    dict(colsample_bynode=0.7, random_state=2),
    dict(colsample_bynode=0.7, random_state=3),
    dict(colsample_bynode=0.7, random_state=4),
    dict(max_depth=3),
    dict(max_depth=5, n_estimators=60, learning_rate=0.1),
    dict(n_estimators=60, learning_rate=0.1),
]
N_MONO = 7
VIEWS = [(0, 0, 0, 0, 0, 0, 0, 0, 0), (1, 0, 0, 0, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0, 0, 0, 0),
         (1, 1, 0, 0, 0, 0, 0, 0, 0), (0, 0, 1, 0, 0, 0, 0, 0, 0), (0, 0, 0, 1, 0, 0, 0, 0, 0),
         (0, 0, 0, 0, 1, 0, 0, 0, 0), (0, 0, 0, 0, 0, 1, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0, 1, 0),
         (0, 0, 0, 0, 0, 0, 0, 0, 1)]


def _new_model(overrides: dict, mono) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(**{**PARAMS, **overrides, "monotone_constraints": mono})


models = []  # (model, view)
for i, kw in enumerate(CONFIGS):
    for view in VIEWS:
        n_feat = 8 + sum(view)
        mono = tuple([1] + [0] * (n_feat - 1)) if i < N_MONO else tuple([0] * n_feat)
        models.append((_new_model(kw, mono), view))

t0 = time.time()
ytr = to_y(train)
FRAMES = {v: (prepare(train, *v), prepare(evald, *v)) for v in VIEWS}
for m, view in models:
    m.fit(FRAMES[view][0], ytr)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    preds = [m.predict_proba(prepare(df, *v))[:, 1] for m, v in models]
    return np.mean(preds, axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
