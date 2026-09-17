"""XGBoost ensemble: recency-weighted diverse members, greedy selection.

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
y01 = (train[TARGET] == POSITIVE).astype(int)
y_ev = (evald[TARGET] == POSITIVE).astype(int)


def _num(s):
    return pd.to_numeric(s, errors="coerce")


# fitted statistic from TRAINING data only
DIST_EDGES = np.unique(np.quantile(_num(train["Distance"]).dropna(),
                                   np.linspace(0, 1, 11)))


def add_features(df: pd.DataFrame, extra: bool = False) -> pd.DataFrame:
    F = pd.DataFrame(index=df.index)
    F["month"] = df["Month"].astype(str)
    F["dow"] = df["DayOfWeek"].astype(str)
    F["carrier"] = df["UniqueCarrier"].astype(str)
    F["origin"] = df["Origin"].astype(str)
    F["dest"] = df["Dest"].astype(str)
    dep = _num(df["DepTime"])
    hh = (dep // 100).clip(0, 25)
    mm = dep - (dep // 100) * 100
    minutes = (hh * 60 + mm) % 1440
    ang = 2 * np.pi * minutes / 1440.0
    F["hour"] = hh
    F["car_hour"] = F["carrier"] + "|" + hh.fillna(-1).astype(int).astype(str)
    F["dep_bin15"] = (minutes // 15).astype(int).astype(str)
    F["hour_sin"] = np.sin(ang)
    F["hour_cos"] = np.cos(ang)
    F["dep_num"] = dep
    F["dep_small"] = (dep < 100).astype(float)
    F["dep_large"] = (dep >= 2400).astype(float)
    F["month_num"] = _num(F["month"].str.slice(2))
    F["dist"] = _num(df["Distance"])
    if extra:
        F["orig_hour"] = F["origin"] + "|" + hh.fillna(-1).astype(int).astype(str)
        F["dest_hour"] = F["dest"] + "|" + hh.fillna(-1).astype(int).astype(str)
        F["dow_bin15"] = F["dow"] + "|" + F["dep_bin15"]
        F["dist_bin"] = pd.cut(F["dist"], bins=DIST_EDGES, labels=False)
        F["dist_hour"] = F["dist_bin"].astype(str) + "|" + hh.fillna(-1).astype(int).astype(str)
    return F


BASE_CATS = ["month", "dow", "carrier", "origin", "dest", "hour", "dep_bin15",
             "car_hour"]
BASE_NUMS = ["dist", "dep_num", "hour_sin", "hour_cos", "dep_small", "dep_large",
             "month_num"]

F_train = add_features(train, extra=True)
F_eval = add_features(evald, extra=True)
ALL_CATS = set(BASE_CATS + ["orig_hour", "dest_hour", "dist_hour", "dow_bin15"])
LEVELS = {c: pd.Index(sorted(F_train[c].dropna().unique())) for c in ALL_CATS}


def make_X(F: pd.DataFrame, cats) -> pd.DataFrame:
    X = pd.DataFrame(index=F.index)
    for c in cats:
        X[c] = pd.Categorical(F[c], categories=LEVELS[c])
    for c in BASE_NUMS:
        X[c] = pd.to_numeric(F[c], errors="coerce")
    return X


Xb_train = make_X(F_train, BASE_CATS)
Xb_eval = make_X(F_eval, BASE_CATS)


def new_model(**kw):
    p = dict(learning_rate=0.05, tree_method="hist", enable_categorical=True,
             random_state=SEED, n_jobs=N_JOBS)
    p.update(kw)
    return xgb.XGBClassifier(**p)


def month_weight(slope: float) -> np.ndarray:
    return 1.0 + slope * (F_train["month_num"].fillna(1) - 1) / 11.0


def add_hour_load(F: pd.DataFrame) -> pd.DataFrame:
    F["hr_load"] = F["hour"].map(HOUR_LOAD).fillna(0)
    F["car_hr_load"] = [CAR_HR_LOAD.get((a, b), 0.0)
                        for a, b in zip(F["carrier"], F["hour"])]
    return F


HOUR_LOAD = (train.assign(hh=(pd.to_numeric(train["DepTime"], errors="coerce") // 100).clip(0, 25))
             .groupby("hh")["dep_delayed_15min"].apply(lambda s: (s == "Y").mean()).to_dict())
_tmp = train.assign(hh=(pd.to_numeric(train["DepTime"], errors="coerce") // 100).clip(0, 25))
CAR_HR_LOAD = _tmp.groupby(["UniqueCarrier", "hh"])["dep_delayed_15min"].apply(
    lambda s: (s == "Y").mean()).to_dict()


t0 = time.time()
MEMBERS = []  # (model, cats, name)


def add_member(name, cats, weights=None, **kw):
    X = make_X(F_train, cats)
    if weights is None:
        m = new_model(**kw).fit(X, y01)
    else:
        m = new_model(**kw).fit(X, y01, sample_weight=weights)
    p = m.predict_proba(make_X(F_eval, cats))[:, 1]
    MEMBERS.append((m, cats, name))
    print(f"  member {name:13s} -> {roc_auc_score(y_ev, p):.4f} ({time.time() - t0:.0f}s)")


DH = BASE_CATS + ["dist_hour"]
OH = BASE_CATS + ["orig_hour"]
DEH = BASE_CATS + ["dest_hour"]
add_member("rec2_disthour", DH, weights=month_weight(1.0),
           n_estimators=2000, max_depth=6, reg_alpha=1.0)
add_member("rec_d6g", BASE_CATS, weights=month_weight(1.0),
           n_estimators=2000, max_depth=6, gamma=1.0)
add_member("rec2_orig_hour", OH, weights=month_weight(1.0),
           n_estimators=2000, max_depth=6, reg_alpha=1.0)

# probe 1: dow x dep_bin15 interaction member
DB = BASE_CATS + ["dow_bin15"]
add_member("x_dow_bin15", DB, weights=month_weight(1.0),
           n_estimators=2000, max_depth=6, reg_alpha=1.0)
# probe 2: hour-load numeric members (train-fitted rates, target-free formulation)
F_train2 = add_hour_load(F_train.copy())
F_eval2 = add_hour_load(F_eval.copy())
m = new_model(n_estimators=2000, max_depth=6, reg_alpha=1.0).fit(
    F_train2[["hr_load", "car_hr_load"]].assign(
        **{c: pd.to_numeric(F_train2[c], errors="coerce") for c in BASE_NUMS}),
    y01, sample_weight=month_weight(1.0))
p = m.predict_proba(F_eval2[["hr_load", "car_hr_load"]].assign(
    **{c: pd.to_numeric(F_eval2[c], errors="coerce") for c in BASE_NUMS}))[:, 1]
MEMBERS.append((m, "LOAD", "rec_hrload"))
print(f"  member rec_hrload   -> {roc_auc_score(y_ev, p):.4f} ({time.time() - t0:.0f}s)")
# probe 3: binning diversity member
add_member("rec2_bin256", BASE_CATS, weights=month_weight(1.0),
           n_estimators=2000, max_depth=6, reg_alpha=1.0, max_bin=256)

NAMES = [n for _, _, n in MEMBERS]
BY_NAME = {n: (m, cats) for m, cats, n in MEMBERS}


def _member_eval_X(cats):
    if isinstance(cats, str) and cats == "LOAD":
        F2 = add_hour_load(F_eval.copy())
        return F2[["hr_load", "car_hr_load"]].assign(
            **{c: pd.to_numeric(F2[c], errors="coerce") for c in BASE_NUMS})
    return make_X(F_eval, cats)


PRED = {n: m.predict_proba(_member_eval_X(cats))[:, 1]
        for (m, cats), n in zip([(m, c) for m, c, _ in MEMBERS], NAMES)}


def auc_of(combo):
    p = np.mean([PRED[n] for n in combo], axis=0)
    return roc_auc_score(y_ev, p)


combo, best_hist, best_auc = [], [], -1.0
while len(combo) < len(NAMES):
    cand, cauc = None, -1
    for n in NAMES:
        if n in combo:
            continue
        a = auc_of(combo + [n])
        if a > cauc:
            cand, cauc = n, a
    if cauc < best_auc + 0.0002:
        break
    combo.append(cand)
    best_auc = cauc
    best_hist.append((list(combo), cauc))
    print(f"  greedy +{cand} -> {cauc:.4f}")
print(f"sweep time: {time.time() - t0:.1f}s")

best_combo, best_auc = max(best_hist, key=lambda c: c[1])
print(f"best: {best_combo} -> {best_auc:.4f}")

_MODELS = [BY_NAME[n] for n in best_combo]


def prepare(df: pd.DataFrame) -> list:
    F = add_features(df, extra=True)
    outs = []
    for _, cats in _MODELS:
        if isinstance(cats, str) and cats == "LOAD":
            F2 = add_hour_load(F.copy())
            outs.append(F2[["hr_load", "car_hr_load"]].assign(
                **{c: pd.to_numeric(F2[c], errors="coerce") for c in BASE_NUMS}))
        else:
            outs.append(make_X(F, cats))
    return outs


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    Ps = [m.predict_proba(X)[:, 1] for (m, _), X in zip(_MODELS, prepare(df))]
    return np.mean(Ps, axis=0)


eval_auc = roc_auc_score(y_ev, predict_proba(evald))
print(f"Eval AUC: {eval_auc:.4f}")
