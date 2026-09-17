"""Airline delay model — v9.

Breakthrough vs v8: multiscale *schedule-congestion* features computed from
train-only counts:
  OHC20/DHC20  log1p # train flights departing/arriving (airport, 20-min bucket)
  OHC40/DHC40  same at 40-min granularity
  OMOV3/DMOV3  moving sum of 20-min counts over 3 buckets (centered)
  OMOV5/DMOV5  moving sum over 5 buckets
Airport schedule banks repeat year over year, so these counts are drift-stable
and give deep trees real structure to fit (2005 -> 2006 transfer).

Extra features (v10):
  CHC20  carrier congestion: log1p # train flights of the same carrier in the
         same 20-min bucket (carrier bank structure)
  DETA8  destination congestion at *estimated arrival* time (DepMinutes + 15
         taxi + Distance/8 minutes, bucketed to 20 min) — arrival-side
         congestion is a physical delay driver
  DepMod position of the departure minute within the hour in 5-min units

Members: very deep (d16-d24) heavily stochastic (subsample 0.85,
colsample_bytree 0.5, colsample_bynode 0.8, reg_lambda 10) XGB models on
BASE+congestion; plus one shallow old-family member mixed in at 20% weight
(measured to help via decorrelation, A/B-validated).

Everything statistical (category levels, congestion counts) is fit on
data/train.csv only. predict_proba(df) works on unseen raw frames.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

TARGET = "dep_delayed_15min"
POS = "Y"
CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
LABEL_COLS = CAT_COLS + ["DepTime", "Distance", TARGET]

train_raw = pd.read_csv("data/train.csv")
eval_raw = pd.read_csv("data/eval.csv")

# ---------------------------------------------------------------- train-only stats
cat_levels = {c: pd.Index(sorted(train_raw[c].astype(str).unique())) for c in CAT_COLS}

_dep_tr = train_raw["DepTime"].astype("int32")
_org_tr = train_raw["Origin"].astype(str)
_dst_tr = train_raw["Dest"].astype(str)
_b20_tr = _dep_tr // 20
_b40_tr = _dep_tr // 40

# (airport, bucket) -> train flight counts, as MultiIndex Series for fast reindex
_cnt_o20 = pd.Series(1, index=pd.MultiIndex.from_arrays([_org_tr, _b20_tr])).groupby(level=[0, 1]).size()
_cnt_d20 = pd.Series(1, index=pd.MultiIndex.from_arrays([_dst_tr, _b20_tr])).groupby(level=[0, 1]).size()
_cnt_o20_dict = _cnt_o20.to_dict()
_cnt_d20_dict = _cnt_d20.to_dict()
_cnt_o40 = (_org_tr + "_" + _b40_tr.astype(str)).value_counts()
_cnt_d40 = (_dst_tr + "_" + _b40_tr.astype(str)).value_counts()
_cnt_c20 = (train_raw["UniqueCarrier"].astype(str) + "_" + _b20_tr.astype(str)).value_counts()

_ETA_MIN_PER_MILE = 8.0
_ETA_TAXI_MIN = 15


def _eta_bucket(dep, dist):
    """Estimated arrival bucket (20-min) from departure time and distance."""
    eta = dep // 100 * 60 + dep % 100 + _ETA_TAXI_MIN + dist / _ETA_MIN_PER_MILE
    return eta.astype("int32") // 20


_cnt_deta = (_dst_tr + "_" + _eta_bucket(_dep_tr, train_raw["Distance"].astype("float32")).astype(str)).value_counts()


def _movsum(tbl_dict, ks, bs, w):
    """Vectorized moving-window train-count sum over centered `w` buckets.

    Missing (k, b) cells contribute 0 — including rows whose (k, b) never
    appears in train (fallback = neighbors only).
    """
    tbl = pd.Series(tbl_dict)
    tbl.index = pd.MultiIndex.from_tuples(tbl.index)
    half = w // 2
    out = np.zeros(len(ks), dtype="float64")
    for off in range(-half, w - half):
        idx = pd.MultiIndex.from_arrays([ks, np.asarray(bs) + off])
        out += tbl.reindex(idx, fill_value=0).to_numpy(dtype="float64")
    return out


def prepare(df):
    """Feature engineering on an unseen raw frame. Statistics from train only."""
    X = pd.DataFrame(index=df.index)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c].astype(str), categories=cat_levels[c])
    dep = df["DepTime"].astype("int32")
    X["DepTime"] = dep
    hour = (dep // 100).clip(0, 24).astype("int16")
    X["Hour"] = hour
    X["Minute"] = (dep % 100).astype("int16")
    X["DepMinutes"] = hour * 60 + (dep % 100).astype("int16")
    X["Distance"] = df["Distance"].astype("float32")

    org = df["Origin"].astype(str)
    dst = df["Dest"].astype(str)
    b20 = (dep // 20).astype("int32")
    b40 = (dep // 40).astype(str)

    X["OHC20"] = (org + "_" + b20.astype(str)).map(_cnt_o20_dict).fillna(0).pipe(np.log1p).astype("float32")
    X["DHC20"] = (dst + "_" + b20.astype(str)).map(_cnt_d20_dict).fillna(0).pipe(np.log1p).astype("float32")
    X["OHC40"] = (org + "_" + b40).map(_cnt_o40).fillna(0).pipe(np.log1p).astype("float32")
    X["DHC40"] = (dst + "_" + b40).map(_cnt_d40).fillna(0).pipe(np.log1p).astype("float32")
    X["OMOV3"] = np.log1p(_movsum(_cnt_o20_dict, org.to_numpy(), b20.to_numpy(), 3)).astype("float32")
    X["DMOV3"] = np.log1p(_movsum(_cnt_d20_dict, dst.to_numpy(), b20.to_numpy(), 3)).astype("float32")
    X["OMOV5"] = np.log1p(_movsum(_cnt_o20_dict, org.to_numpy(), b20.to_numpy(), 5)).astype("float32")
    X["DMOV5"] = np.log1p(_movsum(_cnt_d20_dict, dst.to_numpy(), b20.to_numpy(), 5)).astype("float32")
    X["CHC20"] = (df["UniqueCarrier"].astype(str) + "_" + b20.astype(str)).map(_cnt_c20).fillna(0).pipe(np.log1p).astype("float32")
    X["DETA8"] = (dst + "_" + _eta_bucket(dep, X["Distance"]).astype(str)).map(_cnt_deta).fillna(0).pipe(np.log1p).astype("float32")
    X["DepMod"] = ((dep % 100) // 5).astype("int16")
    return X


BASE = ["DayOfWeek", "UniqueCarrier", "Origin", "Dest", "DepTime", "Hour",
        "Minute", "DepMinutes", "Distance"]
ALLB = BASE + ["OHC20", "DHC20", "OHC40", "DHC40", "OMOV3", "DMOV3", "OMOV5", "DMOV5",
              "CHC20", "DETA8", "DepMod"]

_COMMON = dict(learning_rate=0.05, reg_lambda=10, tree_method="hist",
               enable_categorical=True, n_jobs=4, colsample_bynode=0.8)

# deep stochastic members on the congestion base
DEEP_SPECS = [
    dict(name="d20c50n140", max_depth=20, subsample=0.85, colsample_bytree=0.5, n_estimators=140, random_state=42),
    dict(name="d20c50n200s7", max_depth=20, subsample=0.85, colsample_bytree=0.5, n_estimators=200, random_state=7),
    dict(name="d24c50n200", max_depth=24, subsample=0.85, colsample_bytree=0.5, n_estimators=200, random_state=42),
    dict(name="d16c50n250", max_depth=16, subsample=0.85, colsample_bytree=0.5, n_estimators=250, random_state=42),
]

X_train = prepare(train_raw)
y_train = (train_raw[TARGET] == POS).astype(int).to_numpy()

deep_models = []
for spec in DEEP_SPECS:
    kw = dict(_COMMON)
    kw.update({k: v for k, v in spec.items() if k != "name"})
    m = xgb.XGBClassifier(**kw)
    m.fit(X_train[ALLB], y_train)
    deep_models.append(m)

# old-family member (shallow, no congestion features) — decorrelated, mixed in small
old_model = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                              reg_lambda=1, tree_method="hist", enable_categorical=True,
                              n_jobs=4, colsample_bynode=0.8, random_state=1)
old_model.fit(X_train[BASE], y_train)

OLD_W = 1.0  # old-family weight: 20% of total


def predict_proba(df):
    """Probability of delay for raw rows (works on unseen data)."""
    X = prepare(df)
    p = np.zeros(len(df), dtype="float64")
    for m in deep_models:
        p += m.predict_proba(X[ALLB])[:, 1]
    p += OLD_W * old_model.predict_proba(X[BASE])[:, 1]
    return p / (len(deep_models) + OLD_W)


# ------------------------------------------------------------------ self-evaluation
if len(eval_raw):
    _pred = predict_proba(eval_raw)
    _y = (eval_raw[TARGET] == POS).astype(int).to_numpy()
    print(f"Eval AUC: {roc_auc_score(_y, _pred):.4f}")
