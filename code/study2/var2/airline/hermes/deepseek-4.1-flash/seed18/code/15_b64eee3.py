"""XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

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

RAW_COLS = [c for c in train.columns if c not in ID_COLS + [TARGET]]
OBJ_COLS = [c for c in RAW_COLS if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]

# --- fitted state (training data only) ----------------------------------------
# string/categorical columns -> integer codes; levels fixed from train
CAT_LEVELS = {}
for c in RAW_COLS:
    if c in OBJ_COLS:
        CAT_LEVELS[c] = pd.Index(sorted(train[c].dropna().astype(str).unique()))

# --- schedule-structure tables (fit on training data only) --------------------
N_TR = float(len(train))
_tr_carrier = train["UniqueCarrier"].astype(str)
_tr_origin = train["Origin"].astype(str)
_tr_dest = train["Dest"].astype(str)
_tr_hour = (pd.to_numeric(train["DepTime"], errors="coerce") // 100).clip(upper=23).fillna(-1).astype(int)
_tr_route = _tr_origin + "_" + _tr_dest
_tr_carr_orig = _tr_carrier + "_" + _tr_origin
_tr_orig_hour = _tr_origin + "_" + _tr_hour.astype(str)


def _cnt(s: pd.Series) -> pd.Series:
    return s.value_counts()


_carrier_n, _origin_n, _dest_n, _route_n = _cnt(_tr_carrier), _cnt(_tr_origin), _cnt(_tr_dest), _cnt(_tr_route)
_carr_orig_n = _cnt(_tr_carr_orig)
_orig_hour_n = _cnt(_tr_orig_hour)
_dest_hour_n = _cnt(_tr_dest + "_" + _tr_hour.astype(str))
# carrier's share of an airport's departures in a given hour (hub x peak interaction)
_co_hour_df = pd.DataFrame({"key": (_tr_carrier + "|" + _tr_origin + "|" + _tr_hour.astype(str)).value_counts().index})
_co_hour_df["n"] = (_tr_carrier + "|" + _tr_origin + "|" + _tr_hour.astype(str)).value_counts().to_numpy()
_co_hour_df["oh"] = [k.split("|", 1)[1] for k in _co_hour_df["key"]]
CO_HOUR_SHARE = (_co_hour_df.set_index("key")["n"] / _co_hour_df["oh"].map(_co_hour_df.groupby("oh")["n"].sum())).to_dict()

# frequency tables: key -> share of training rows
FREQ = {
    "origin": (_origin_n / N_TR).to_dict(),
    "dest": (_dest_n / N_TR).to_dict(),
    "carrier": (_carrier_n / N_TR).to_dict(),
    "route": (_route_n / N_TR).to_dict(),
}
# hub structure: how concentrated a carrier is at an airport
_carr_orig_split = _carr_orig_n.index.str.split("_")
_co_origin_key = [s[1] for s in _carr_orig_split]  # "UA_TPA" -> TPA (airport)
_co_carrier_key = [s[0] for s in _carr_orig_split]  # "UA_TPA" -> UA  (carrier)
CO_SHARE_ORIGIN = (_carr_orig_n / _origin_n.reindex(_co_origin_key).to_numpy()).to_dict()
CO_SHARE_CARRIER = (_carr_orig_n / _carrier_n.reindex(_co_carrier_key).to_numpy()).to_dict()
ORIG_HOUR_SHARE = (_orig_hour_n / _origin_n.reindex([s.rsplit("_", 1)[0] for s in _orig_hour_n.index]).to_numpy()).to_dict()
DEST_HOUR_SHARE = (_dest_hour_n / _dest_n.reindex([s.rsplit("_", 1)[0] for s in _dest_hour_n.index]).to_numpy()).to_dict()

# cumulative share of an airport's daily departures scheduled up to a given hour:
# a congestion proxy that is pure schedule structure (stable across years)
_oh = pd.DataFrame({"key": _orig_hour_n.index, "n": _orig_hour_n.to_numpy()})
_oh["origin"] = [k.rsplit("_", 1)[0] for k in _oh["key"]]
_oh["hour"] = [int(k.rsplit("_", 1)[1]) for k in _oh["key"]]
_oh = _oh.sort_values(["origin", "hour"])
CUM_ORIG_HOUR = (
    _oh.assign(cum=_oh.groupby("origin")["n"].cumsum() / _oh.groupby("origin")["n"].transform("sum"))
    .set_index("key")["cum"]
    .to_dict()
)


def _cum_by_prefix(cnt: pd.Series) -> dict:
    """key '<K>_<hour>' -> cumulative share of K's volume up to that hour."""
    d = pd.DataFrame({"key": cnt.index, "n": cnt.to_numpy()})
    d["pre"] = [k.rsplit("_", 1)[0] for k in d["key"]]
    d["hour"] = [int(k.rsplit("_", 1)[1]) for k in d["key"]]
    d = d.sort_values(["pre", "hour"])
    return (d.assign(cum=d.groupby("pre")["n"].cumsum() / d.groupby("pre")["n"].transform("sum"))
            .set_index("key")["cum"].to_dict())


CUM_DEST_HOUR = _cum_by_prefix(_dest_hour_n)


def _to_num(s: pd.Series) -> pd.Series:
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        return pd.to_numeric(s.astype(str).str.replace("c-", "", regex=False), errors="coerce")
    return pd.to_numeric(s, errors="coerce")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Raw airflow-style row -> model matrix. All feature engineering lives here."""
    X = pd.DataFrame(index=df.index)

    dow = _to_num(df["DayOfWeek"])
    X["dayofweek"] = dow
    X["is_weekend"] = (dow >= 6).astype(float)

    dt = _to_num(df["DepTime"])
    hour = (dt // 100).clip(upper=23).fillna(-1)
    minute = (dt % 100).fillna(0)
    X["dep_hour"] = hour
    X["dep_minute"] = minute
    tod = hour * 60 + minute
    X["dep_tod"] = tod
    X["dep_tod_sin"] = np.sin(2 * np.pi * tod / 1440.0)
    X["dep_tod_cos"] = np.cos(2 * np.pi * tod / 1440.0)

    X["distance"] = pd.to_numeric(df["Distance"], errors="coerce")

    carrier = df["UniqueCarrier"].astype(str)
    origin = df["Origin"].astype(str)
    dest = df["Dest"].astype(str)
    X["carrier"] = pd.Categorical(carrier, categories=CAT_LEVELS["UniqueCarrier"])

    # schedule-structure lookups (built from train; unseen keys -> 0)
    route = origin + "_" + dest
    carr_orig = carrier + "_" + origin
    orig_hour = origin + "_" + hour.astype(int).astype(str)
    X["freq_origin"] = pd.Series(origin.values).map(FREQ["origin"]).fillna(0.0).to_numpy()
    X["freq_dest"] = pd.Series(dest.values).map(FREQ["dest"]).fillna(0.0).to_numpy()
    X["freq_carrier"] = pd.Series(carrier.values).map(FREQ["carrier"]).fillna(0.0).to_numpy()
    X["freq_route"] = pd.Series(route.values).map(FREQ["route"]).fillna(0.0).to_numpy()
    X["co_share_origin"] = pd.Series(carr_orig.values).map(CO_SHARE_ORIGIN).fillna(0.0).to_numpy()
    X["co_share_carrier"] = pd.Series(carr_orig.values).map(CO_SHARE_CARRIER).fillna(0.0).to_numpy()
    X["orig_hour_share"] = pd.Series(orig_hour.values).map(ORIG_HOUR_SHARE).fillna(0.0).to_numpy()
    X["orig_hour_cum"] = pd.Series(orig_hour.values).map(CUM_ORIG_HOUR).fillna(0.0).to_numpy()
    dest_hour = dest + "_" + hour.astype(int).astype(str)
    co_hour = carrier + "|" + origin + "|" + hour.astype(int).astype(str)
    X["dest_hour_share"] = pd.Series(dest_hour.values).map(DEST_HOUR_SHARE).fillna(0.0).to_numpy()
    X["co_hour_share"] = pd.Series(co_hour.values).map(CO_HOUR_SHARE).fillna(0.0).to_numpy()
    X["dest_hour_cum"] = pd.Series(dest_hour.values).map(CUM_DEST_HOUR).fillna(0.0).to_numpy()

    return X


CAT_COLS = ["carrier", "origin", "dest"]


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model: small ensemble of XGBoost variants (averaged probabilities) --------
MODEL_PARAMS = [
    dict(n_estimators=600, max_depth=6, learning_rate=0.04, subsample=0.9, colsample_bytree=0.8, min_child_weight=3, random_state=1),
    dict(n_estimators=600, max_depth=7, learning_rate=0.04, subsample=0.8, colsample_bytree=0.7, min_child_weight=3, random_state=7),
    dict(n_estimators=600, max_depth=8, learning_rate=0.04, subsample=0.8, colsample_bytree=0.7, min_child_weight=5, random_state=13),
    dict(n_estimators=600, max_depth=9, learning_rate=0.04, subsample=0.8, colsample_bytree=0.8, min_child_weight=5, random_state=42),
    dict(n_estimators=600, max_depth=9, learning_rate=0.04, subsample=0.9, colsample_bytree=0.5, min_child_weight=10, random_state=99),
    dict(n_estimators=600, max_depth=10, learning_rate=0.04, subsample=0.7, colsample_bytree=0.8, min_child_weight=8, random_state=2024),
    dict(n_estimators=600, max_depth=0, max_leaves=255, grow_policy="lossguide", learning_rate=0.04, subsample=0.8, colsample_bytree=0.7, min_child_weight=10, random_state=11),
    dict(n_estimators=600, max_depth=9, learning_rate=0.04, subsample=0.8, colsample_bytree=0.8, min_child_weight=5, max_bin=512, random_state=555),
]
COMMON = dict(tree_method="hist", enable_categorical=True, n_jobs=N_JOBS)

X_train = prepare(train)
Y_train = to_y(train)

t0 = time.time()
models = []
for p in MODEL_PARAMS:
    m = xgb.XGBClassifier(**{**COMMON, **p})
    m.fit(X_train, Y_train)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")

_imp = np.mean([m.feature_importances_ for m in models], axis=0)
for _n, _v in sorted(zip(X_train.columns, _imp), key=lambda t: -t[1])[:20]:
    print(f"  imp {_n:20s} {_v:.4f}")
p_tr = np.mean([m.predict_proba(X_train)[:, 1] for m in models], axis=0)
print(f"Train AUC: {roc_auc_score(Y_train, p_tr):.4f}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
