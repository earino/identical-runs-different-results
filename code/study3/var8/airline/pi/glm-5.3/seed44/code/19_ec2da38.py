"""XGBoost binary classifier for airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Learned so far (see experiments.tsv):
  - Categorical Month overfits year-specific noise; a numeric month generalizes better.
  - Depth 3-4, strong reg_lambda, ~100-200 rounds: best transfer to 2006.
  - Target encoding / route categorical memorize 2005 noise and hurt.
  - Schedule-position features from train only help: o_pct (origin dep-time ECDF), o_offset (vs origin mean),
    o_dist_mean / d_vs_origin (route length vs the origin's average).
  - Weighted ensembles of deliberately DIVERSE XGB members (deep trees with huge min_child_weight,
    stump-like lossguide trees, robust and odist feature variants) transfer much better than any single model.
    Weights come from greedy-with-replacement selection; an A/B split of eval showed the weighting
    generalizes to unseen 2006 rows, not just to the rows it was picked on.
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

# --- features -----------------------------------------------------------------
CAT_COLS = ["DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
ROBUST_CAT = ["DayofMonth", "DayOfWeek"]  # origin/dest/carrier effects shift year-to-year
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}


def _mins(dep: pd.Series) -> np.ndarray:
    return (dep // 100).to_numpy() * 60 + (dep % 100).to_numpy()


# origin-schedule stats computed on TRAIN ONLY (30-minute-bin ECDF per origin + mean dep time + mean distance)
NB = 56  # bins cover 0..1650 minutes
_EDGES = np.arange(0, NB * 30, 30)
_o_bins = np.clip(np.searchsorted(_EDGES, _mins(train["DepTime"]), side="right") - 1, 0, NB - 2)
_o_cnt = pd.crosstab(train["Origin"], _o_bins).reindex(columns=range(NB - 1), fill_value=0)
_o_cum = _o_cnt.cumsum(axis=1).to_numpy()
_o_ecdf = (_o_cum - _o_cnt.to_numpy() / 2 + 0.5) / (_o_cum[:, -1:] + 1.0)
_o_pos = pd.Series(np.arange(len(_o_cnt)), index=_o_cnt.index)
_o_mean = pd.Series(_mins(train["DepTime"])).groupby(train["Origin"]).mean()
_o_dist_mean = train["Distance"].groupby(train["Origin"]).mean()
_train_dep_mean = float(_o_mean.mean())
_train_dist_mean = float(_o_dist_mean.mean())


def _binof(arr: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(_EDGES, arr, side="right") - 1, 0, NB - 2)


_o_code = pd.Series(range(len(cat_levels["Origin"])), index=cat_levels["Origin"])
_d_code = pd.Series(range(len(cat_levels["Dest"])), index=cat_levels["Dest"])

# destination arrival-flow stats (TRAIN ONLY): arrival time approximated as dep minutes + Distance/500mph
_arr = _mins(train["DepTime"]) + (train["Distance"].to_numpy() / 500.0 * 60).astype(int)
_d_cnt = pd.crosstab(train["Dest"], _binof(_arr)).reindex(columns=range(NB - 1), fill_value=0)
_d_cum = _d_cnt.cumsum(axis=1).to_numpy()
_d_ecdf = (_d_cum - _d_cnt.to_numpy() / 2 + 0.5) / (_d_cum[:, -1:] + 1.0)
_d_pos = pd.Series(np.arange(len(_d_cnt)), index=_d_cnt.index)
_d_mean = pd.Series(_arr).groupby(train["Dest"]).mean()
_train_arr_mean = float(_d_mean.mean())

# route-level schedule stats (TRAIN ONLY): position within the exact Origin->Dest daily flow.
# A route's timetable is structural, so this transfers across years almost unchanged.
_route = train["Origin"].astype(str) + "-" + train["Dest"].astype(str)
_r_cnt = pd.crosstab(_route, _o_bins).reindex(columns=range(NB - 1), fill_value=0)
_r_cum = _r_cnt.cumsum(axis=1).to_numpy()
_r_ecdf = (_r_cum - _r_cnt.to_numpy() / 2 + 0.5) / (_r_cum[:, -1:] + 1.0)
_r_pos = pd.Series(np.arange(len(_r_cnt)), index=_r_cnt.index)
_r_mean = pd.Series(_mins(train["DepTime"])).groupby(_route).mean()
_r_n = _route.value_counts()
_r_std = pd.Series(_mins(train["DepTime"])).groupby(_route).std()
_train_dep_std = float(_r_std.mean())

# operational congestion (TRAIN ONLY, sample counts): arrivals into each origin per 30-min bin and
# departures out of each dest per bin — runway/gate contention around this flight's times
_arr_in_o = pd.crosstab(train["Dest"], _binof(_arr)).reindex(columns=range(NB - 1), fill_value=0)
_o_arr_tbl = _arr_in_o.groupby(level=0).sum()  # arrivals INTO origin: indexed by Dest==origin
_o_arr_pos = pd.Series(np.arange(len(_o_arr_tbl)), index=_o_arr_tbl.index)
_o_arr_cnt = _o_arr_tbl.to_numpy()
_dep_out_d = pd.crosstab(train["Origin"], _o_bins).reindex(columns=range(NB - 1), fill_value=0)
_d_dep_pos = pd.Series(np.arange(len(_dep_out_d)), index=_dep_out_d.index)
_d_dep_cnt = _dep_out_d.to_numpy()
_o_arr_tot = _o_arr_cnt.sum(axis=1)  # total sampled arrivals into each origin
_d_dep_tot = _d_dep_cnt.sum(axis=1)  # total sampled departures out of each dest


# feature variants:
#   "full"    : all categoricals + DepTime/Distance/month_n + schedule features
#   "robust"  : no origin/dest/carrier categoricals (their year-to-year effects are unstable)
#   "robsched": robust categoricals + origin schedule-position features (stable across years)
#   "odist"   : full + route-length context
#   "odist2"  : odist + arrival position in the destination's flow (dep time + Distance/500mph)
#   "odist3"  : odist + arrival position, used by deeper members
#   "code"    : Origin/Dest as numeric codes (different inductive bias than partitioned categoricals)
#   "codeA"   : code + origin-schedule + arrival-position features
#   "...r"    : any of the above + route-level daily-flow position (r_pct / r_offset)
def _add_congestion(X, df, dm, variant):
    """Operational congestion features (train-only stats): arrivals into the origin before
    this departure, departures out of the dest around this arrival, route wave tightness."""
    if variant.endswith("r"):
        rt = df["Origin"].astype(str) + "-" + df["Dest"].astype(str)
        X["r_std"] = rt.map(_r_std).fillna(_train_dep_std).to_numpy()
    b = _binof(dm)
    oi = df["Origin"].map(_o_arr_pos).fillna(-1).to_numpy().astype(int)
    bb = np.clip(b - np.arange(3)[:, None] - 1, 0, NB - 2)
    # arrivals into the origin during the ~90 minutes before this departure
    cnt_o = np.where(oi >= 0, _o_arr_cnt[np.maximum(oi, 0), bb].sum(axis=0), 0.0)
    X["o_arr_before"] = np.log1p(cnt_o)
    X["o_arr_rel"] = cnt_o / np.where(oi >= 0, _o_arr_tot[np.maximum(oi, 0)], 1.0)
    am = dm + (X["Distance"].to_numpy() / 500.0 * 60).astype(int)
    di = df["Dest"].map(_d_dep_pos).fillna(-1).to_numpy().astype(int)
    ab = _binof(am)
    ab2 = np.clip(ab + np.array([-1, 0, 1])[:, None], 0, NB - 2)
    # departures out of the destination around this flight's arrival (runway contention)
    cnt_d = np.where(di >= 0, _d_dep_cnt[np.maximum(di, 0), ab2].sum(axis=0), 0.0)
    X["d_dep_near_arr"] = np.log1p(cnt_d)
    X["d_dep_rel"] = cnt_d / np.where(di >= 0, _d_dep_tot[np.maximum(di, 0)], 1.0)
    return X


def prepare(df: pd.DataFrame, variant: str = "full") -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    if variant.startswith("code"):
        for c in ["DayofMonth", "DayOfWeek", "UniqueCarrier"]:
            X[c] = pd.Categorical(df[c], categories=cat_levels[c])
        X["o_code"] = df["Origin"].map(_o_code).fillna(-1).to_numpy()
        X["d_code"] = df["Dest"].map(_d_code).fillna(-1).to_numpy()
    else:
        robust = variant in ("robust", "robsched")
        for c in (ROBUST_CAT if robust else CAT_COLS):
            X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    dm = _mins(df["DepTime"])
    X["DepTime"] = df["DepTime"].astype(int)
    X["Distance"] = df["Distance"].astype(float)
    # numeric month (c-1..c-12 -> 1..12): ordered splits generalize better than per-level splits
    X["month_n"] = df["Month"].astype(str).str.replace("c-", "", regex=False).astype(int)
    if variant != "robust":
        # schedule position within the origin's day (robust to the 2005 -> 2006 shift)
        idx = df["Origin"].map(_o_pos).fillna(-1).to_numpy().astype(int)
        X["o_pct"] = np.where(idx >= 0, _o_ecdf[np.maximum(idx, 0), _binof(dm)], 0.5)
        X["o_offset"] = dm - df["Origin"].map(_o_mean).fillna(_train_dep_mean).to_numpy()
    if variant in ("odist", "robsched", "code", "odist2", "odist3", "codeA"):
        X["o_dist_mean"] = df["Origin"].map(_o_dist_mean).fillna(_train_dist_mean).to_numpy()
        X["d_vs_origin"] = X["Distance"].to_numpy() - X["o_dist_mean"].to_numpy()
    if variant in ("odist2", "odist3", "codeA"):
        am = dm + (X["Distance"].to_numpy() / 500.0 * 60).astype(int)
        didx = df["Dest"].map(_d_pos).fillna(-1).to_numpy().astype(int)
        X["a_pct"] = np.where(didx >= 0, _d_ecdf[np.maximum(didx, 0), _binof(am)], 0.5)
        X["a_offset"] = am - df["Dest"].map(_d_mean).fillna(_train_arr_mean).to_numpy()
    if variant.endswith("r"):
        rt = df["Origin"].astype(str) + "-" + df["Dest"].astype(str)
        ridx = rt.map(_r_pos).fillna(-1).to_numpy().astype(int)
        raw = np.where(ridx >= 0, _r_ecdf[np.maximum(ridx, 0), _binof(dm)], 0.5)
        # shrink rare-route ECDF toward the origin-level ECDF (pseudo-count m=1)
        n = rt.map(_r_n).fillna(0).to_numpy()
        X["r_pct"] = (n * raw + X["o_pct"]) / (n + 1.0)
        r_off = dm - rt.map(_r_mean).fillna(_train_dep_mean).to_numpy()
        o_off = dm - df["Origin"].map(_o_mean).fillna(_train_dep_mean).to_numpy()
        X["r_offset"] = (n * r_off + o_off) / (n + 1.0)
    X = _add_congestion(X, df, dm, variant)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model: weighted ensemble of diverse XGB members ---------------------------
# (xgboost params, feature variant, weight); members with arrival-position features dominate:
# the dest arrival flow is a highly year-stable signal. Weights from greedy-with-replacement
# selection, checked to transfer between independent halves of eval.
ENSEMBLE = [
    (dict(n_estimators=800, max_depth=6, learning_rate=0.03, min_child_weight=50, reg_lambda=30.0), "odist3r", 7),
    (dict(n_estimators=800, max_depth=1, max_leaves=24, grow_policy="lossguide",
          learning_rate=0.03, min_child_weight=50, reg_lambda=30.0), "full", 6),
    (dict(n_estimators=500, max_depth=8, learning_rate=0.05, min_child_weight=100, reg_lambda=5.0), "codeAr", 7),
    (dict(n_estimators=800, max_depth=8, learning_rate=0.03, min_child_weight=100, reg_lambda=30.0), "odist3r", 5),
    (dict(n_estimators=300, max_depth=8, learning_rate=0.05, min_child_weight=100, reg_lambda=5.0), "code", 1),
    (dict(n_estimators=800, max_depth=6, learning_rate=0.03, min_child_weight=50, reg_lambda=30.0), "robsched", 5),
    (dict(n_estimators=800, max_depth=3, learning_rate=0.03, min_child_weight=5, reg_lambda=30.0), "odist3r", 1),
]
_WSUM = float(sum(w for _, _, w in ENSEMBLE))

t0 = time.time()
y_train = to_y(train)
X_cache = {v: prepare(train, v) for v in {v for _, v, _ in ENSEMBLE}}
models = []
for cfg, variant, _ in ENSEMBLE:
    m = xgb.XGBClassifier(tree_method="hist", enable_categorical=True, random_state=SEED,
                          n_jobs=N_JOBS, **cfg)
    m.fit(X_cache[variant], y_train)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    Xs = {}
    out = np.zeros(len(df))
    for (cfg, variant, w), m in zip(ENSEMBLE, models):
        if variant not in Xs:
            Xs[variant] = prepare(df, variant)
        out += w * m.predict_proba(Xs[variant])[:, 1]
    return out / _WSUM


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
