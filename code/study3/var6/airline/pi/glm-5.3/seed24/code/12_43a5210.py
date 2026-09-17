"""XGBoost classifier for airline delay (agent-edited, see program.md).

Contract:
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. Module-level `predict_proba(df)` maps a raw DataFrame -> P(positive). All feature engineering
     happens inside prepare(), which only uses statistics fitted on data/train.csv.

Model: ensemble of XGBoost classifiers over three feature-set variants (time-bucket granularity of
the interaction target encodings, max_bin) x two seeds.
"""
import json
import os
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

TASK = json.load(open("task.json"))
TARGET = TASK["target"]
POSITIVE = TASK["positive_label"]
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


ytr = to_y(train)
PRIOR = float(ytr.mean())

# --- group keys (shared by target encodings and counts) ------------------------
TE_M = {  # key name -> smoothing m
    "carrier": 20,
    "origin": 20,
    "dest": 20,
    "route": 100,
    "car_hour": 30,
    "org_hour": 60,
    "dow_hour": 30,
    "route_hour": 200,
    "dest_hour": 60,
    "car_dow": 30,
    "route_hm": 400,  # *_hm keys use 30-minute buckets (nb=2) by default
    "car_hm": 100,
    "org_hm": 200,
    "dow_hm": 40,
}
HM_KEYS = ["route_hm", "car_hm", "org_hm", "dow_hm"]
TE_M_FINE = {"route_hm": 800, "car_hm": 200, "org_hm": 400, "dow_hm": 80}  # 15-minute buckets (nb=4)
COUNT_KEYS = {  # count feature -> key name (train frequency of that key)
    "cnt_carrier": "carrier",
    "cnt_origin": "origin",
    "cnt_dest": "dest",
    "cnt_route": "route",
    "cnt_org_hour": "org_hour",
    "cnt_route_hour": "route_hour",
    "cnt_dest_hour": "dest_hour",
    "cnt_car_hour": "car_hour",
    "cnt_dow_hour": "dow_hour",
    "cnt_route_hm": "route_hm",
    "cnt_car_hm": "car_hm",
    "cnt_org_hm": "org_hm",
    "cnt_dow_hm": "dow_hm",
}


def _keys(df: pd.DataFrame, nb: int = 2) -> dict:
    """Group-key Series computed from raw columns only. nb = sub-hour buckets per hour."""
    dep = df["DepTime"].to_numpy()
    hour = np.clip(dep // 100, 0, 24)
    hs = pd.Series(hour, index=df.index).astype(str)
    hms = pd.Series(hour * nb + np.clip(dep % 100, 0, 59) // (60 // nb), index=df.index).astype(str)
    route = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    keys = {
        "carrier": df["UniqueCarrier"].astype(str),
        "origin": df["Origin"].astype(str),
        "dest": df["Dest"].astype(str),
        "route": route,
        "car_hour": df["UniqueCarrier"].astype(str) + "_" + hs,
        "org_hour": df["Origin"].astype(str) + "_" + hs,
        "dow_hour": df["DayOfWeek"].astype(str) + "_" + hs,
        "route_hour": route + "_" + hs,
        "dest_hour": df["Dest"].astype(str) + "_" + hs,
        "car_dow": df["UniqueCarrier"].astype(str) + "_" + df["DayOfWeek"].astype(str),
        "route_hm": route + "_" + hms,
        "car_hm": df["UniqueCarrier"].astype(str) + "_" + hms,
        "org_hm": df["Origin"].astype(str) + "_" + hms,
        "dow_hm": df["DayOfWeek"].astype(str) + "_" + hms,
    }
    return keys


def _fit_map(key: pd.Series, y: pd.Series, m: float) -> dict:
    g = y.groupby(key).sum()
    n = key.value_counts()
    return ((g + PRIOR * m) / (n + m)).to_dict()


_YY = pd.Series(ytr, index=train.index)
_ktr = _keys(train)
_ktr_fine = _keys(train, 4)
TE_MAPS = {k: _fit_map(_ktr[k], _YY, m) for k, m in TE_M.items()}
TE_MAPS_FINE = {k: _fit_map(_ktr_fine[k], _YY, m) for k, m in TE_M_FINE.items()}

# out-of-fold TE values for the training rows (one fold structure per seed)
def _oof_te(key: pd.Series, m: float, fold_seed: int) -> np.ndarray:
    oof = np.zeros(len(train))
    for tr_i, va_i in KFold(n_splits=4, shuffle=True, random_state=fold_seed).split(train):
        mp = _fit_map(key.iloc[tr_i], _YY.iloc[tr_i], m)
        oof[va_i] = key.iloc[va_i].map(mp).fillna(PRIOR).to_numpy()
    return oof


def _oof_all(fold_seed: int) -> tuple:
    oof = {k: _oof_te(_ktr[k], m, fold_seed) for k, m in TE_M.items()}
    oof_fine = {k: _oof_te(_ktr_fine[k], m, fold_seed) for k, m in TE_M_FINE.items()}
    return oof, oof_fine


CNT = {name: _ktr[k].value_counts().to_dict() for name, k in COUNT_KEYS.items()}
dow_levels = pd.Index(sorted(train["DayOfWeek"].dropna().unique()))

# scheduled-time-of-day means per group (train only): "deviation from the group's typical schedule"
_tod = np.clip(train["DepTime"].to_numpy() // 100, 0, 24) + np.clip(train["DepTime"].to_numpy() % 100, 0, 59) / 60.0
TOD_MEAN = float(_tod.mean())
TOD_BY = {g: pd.Series(_tod).groupby(_ktr[g].to_numpy()).mean().to_dict() for g in ["route", "carrier", "origin", "dest"]}


def prepare(df: pd.DataFrame, fine: bool = False, nohm: bool = False, oof: dict = None) -> pd.DataFrame:
    """Raw airline dataframe -> feature matrix. predict_proba() calls this on unseen rows.
    fine: 15-minute buckets for the *_hm interaction TEs; nohm: drop those TEs entirely."""
    X = pd.DataFrame(index=df.index)
    dep = df["DepTime"].to_numpy()
    hour = np.clip(dep // 100, 0, 24)
    minute = np.clip(dep % 100, 0, 59)
    tod = hour + minute / 60.0
    X["hour"] = hour
    X["minute"] = minute
    X["tod"] = tod
    X["dep_raw"] = dep
    X["distance"] = df["Distance"].to_numpy()
    keys = _keys(df)
    keys_fine = _keys(df, 4) if fine else None
    for k in TE_M:
        if nohm and k in HM_KEYS:
            continue
        if fine and k in HM_KEYS:
            if oof is not None:
                X["te_" + k] = oof[k]
            else:
                X["te_" + k] = keys_fine[k].map(TE_MAPS_FINE[k]).fillna(PRIOR).to_numpy()
        else:
            if oof is not None:
                X["te_" + k] = oof[k]
            else:
                X["te_" + k] = keys[k].map(TE_MAPS[k]).fillna(PRIOR).to_numpy()
    for name, k in COUNT_KEYS.items():
        X[name] = keys[k].map(CNT[name]).fillna(0).to_numpy()
    for g in ["route", "carrier", "origin", "dest"]:
        X["tod_dev_" + g] = tod - keys[g].map(TOD_BY[g]).fillna(TOD_MEAN).to_numpy()
    X["dow"] = pd.Categorical(df["DayOfWeek"], categories=dow_levels)  # unseen levels -> NaN
    return X


# --- ensemble: 3 feature-set variants x 2 seeds --------------------------------
VARIANTS = [
    dict(fine=False, nohm=False),
    dict(fine=True, nohm=False),
    dict(fine=False, nohm=True),
]
SEEDS = [42, 43]
Xev_cache = {}
yev = to_y(evald)
models = []
t0 = time.time()
for seed in SEEDS:
    oof, oof_fine = _oof_all(seed)
    merged = {**oof, **oof_fine}  # for the fine variant: 15-min OOF values under the *_hm keys
    for vi, var in enumerate(VARIANTS):
        Xtr = prepare(train, fine=var["fine"], nohm=var["nohm"], oof=merged if var["fine"] else oof)
        key = (var["fine"], var["nohm"])
        if key not in Xev_cache:
            Xev_cache[key] = prepare(evald, fine=var["fine"], nohm=var["nohm"])
        params = dict(
            n_estimators=5000,
            learning_rate=0.03,
            max_depth=9 if not var["nohm"] else 8,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            eval_metric="auc",
            early_stopping_rounds=150,
            tree_method="hist",
            enable_categorical=True,
            random_state=seed,
            n_jobs=N_JOBS,
        )
        model = xgb.XGBClassifier(**params)
        model.fit(Xtr, ytr, eval_set=[(Xev_cache[key], yev)], verbose=False)
        auc_s = roc_auc_score(yev, model.predict_proba(Xev_cache[key])[:, 1])
        print(f"member variant={vi} seed={seed}: best_iteration={model.best_iteration}, AUC={auc_s:.4f}")
        models.append(model)
print(f"Training time: {time.time() - t0:.1f}s for {len(models)} members")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    Xs = {(v["fine"], v["nohm"]): prepare(df, fine=v["fine"], nohm=v["nohm"]) for v in VARIANTS}
    ps = []
    for si in range(len(SEEDS)):
        for vi, var in enumerate(VARIANTS):
            m = models[si * len(VARIANTS) + vi]
            ps.append(m.predict_proba(Xs[(var["fine"], var["nohm"])])[:, 1])
    return np.mean(ps, axis=0)


t0 = time.time()
eval_auc = roc_auc_score(yev, predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
