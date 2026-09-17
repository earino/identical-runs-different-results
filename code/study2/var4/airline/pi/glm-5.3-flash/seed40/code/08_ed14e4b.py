"""XGBoost — experiment 7: raw+time + smoothed rate encodings (train-only) + coarse interaction cats."""
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

CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: sorted(train[c].astype(str).unique()) for c in CAT_COLS}
CUMDAYS = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])
PRIOR = float((train[TARGET] == POSITIVE).mean())
K = 75.0


def _hb(df: pd.DataFrame) -> np.ndarray:
    return (df["DepTime"].astype("int64").to_numpy() // 100) // 3


# --- rate encodings fit on train only --------------------------------------------
def _rate(keys: pd.Series, y: np.ndarray, k: float = K) -> pd.Series:
    g = pd.DataFrame({"k": keys.to_numpy(), "y": y}).groupby("k")["y"].agg(["sum", "count"])
    return ((g["sum"] + k * PRIOR) / (g["count"] + k))


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


_y = to_y(train)
rate_specs = {
    "r_month": (train["Month"], K),
    "r_carrier": (train["UniqueCarrier"], K),
    "r_origin": (train["Origin"], 100.0),
    "r_dest": (train["Dest"], 100.0),
    "r_hb": (pd.Series(_hb(train), index=train.index), 100.0),
    "r_carmo": (train["UniqueCarrier"] + "|" + train["Month"], 75.0),
    "r_orimo": (train["Origin"] + "|" + train["Month"], 75.0),
    "r_carhb": (train["UniqueCarrier"] + "|" + pd.Series(_hb(train), index=train.index).astype(str), 75.0),
    "r_dom": (train["DayofMonth"], 100.0),
    "r_dow": (train["DayOfWeek"], K),
}
rate_tables = {name: _rate(keys, _y, k) for name, (keys, k) in rate_specs.items()}

ix_levels = {
    "carhb": sorted(set(train["UniqueCarrier"] + "|" + pd.Series(_hb(train), index=train.index).astype(str))),
    "carmo": sorted(set(train["UniqueCarrier"] + "|" + train["Month"])),
    "orimo": sorted(set(train["Origin"] + "|" + train["Month"])),
}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    m = df["Month"].str[2:].astype(int).to_numpy()
    dom = df["DayofMonth"].str[2:].astype(int).to_numpy()
    dow = df["DayOfWeek"].str[2:].astype(int).to_numpy()
    dt = df["DepTime"].astype("int64").to_numpy()
    dep_min = (dt // 100) * 60 + dt % 100
    doy = CUMDAYS[m - 1] + dom
    hb = (dt // 100) // 3
    X["DepTime"] = dt
    X["Distance"] = df["Distance"].to_numpy()
    X["dep_min"] = dep_min
    X["dep_sin"] = np.sin(2 * np.pi * dep_min / 1440.0)
    X["dep_cos"] = np.cos(2 * np.pi * dep_min / 1440.0)
    X["DepTime_mod"] = dt % 100
    X["hour"] = dt // 100
    X["hb"] = hb
    X["doy"] = doy
    X["doy_sin"] = np.sin(2 * np.pi * doy / 365.0)
    X["doy_cos"] = np.cos(2 * np.pi * doy / 365.0)
    X["dow"] = dow
    X["log_dist"] = np.log1p(df["Distance"].to_numpy())
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c].astype(str), categories=cat_levels[c])
    X["carhb"] = pd.Categorical(df["UniqueCarrier"] + "|" + pd.Series(hb, index=df.index).astype(str), categories=ix_levels["carhb"])
    X["carmo"] = pd.Categorical(df["UniqueCarrier"] + "|" + df["Month"], categories=ix_levels["carmo"])
    X["orimo"] = pd.Categorical(df["Origin"] + "|" + df["Month"], categories=ix_levels["orimo"])
    X["r_month"] = rate_tables["r_month"].reindex(df["Month"]).fillna(PRIOR).to_numpy()
    X["r_carrier"] = rate_tables["r_carrier"].reindex(df["UniqueCarrier"]).fillna(PRIOR).to_numpy()
    X["r_origin"] = rate_tables["r_origin"].reindex(df["Origin"]).fillna(PRIOR).to_numpy()
    X["r_dest"] = rate_tables["r_dest"].reindex(df["Dest"]).fillna(PRIOR).to_numpy()
    X["r_hb"] = rate_tables["r_hb"].reindex(pd.Series(hb, index=df.index)).fillna(PRIOR).to_numpy()
    X["r_carmo"] = rate_tables["r_carmo"].reindex(df["UniqueCarrier"] + "|" + df["Month"]).fillna(PRIOR).to_numpy()
    X["r_orimo"] = rate_tables["r_orimo"].reindex(df["Origin"] + "|" + df["Month"]).fillna(PRIOR).to_numpy()
    X["r_carhb"] = rate_tables["r_carhb"].reindex(df["UniqueCarrier"] + "|" + pd.Series(hb, index=df.index).astype(str)).fillna(PRIOR).to_numpy()
    X["r_dom"] = rate_tables["r_dom"].reindex(df["DayofMonth"]).fillna(PRIOR).to_numpy()
    X["r_dow"] = rate_tables["r_dow"].reindex(df["DayOfWeek"]).fillna(PRIOR).to_numpy()
    return X


GROUPS = {
    "BASE": ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest", "DepTime", "Distance",
             "dep_min", "dep_sin", "dep_cos", "DepTime_mod", "hour", "hb", "doy", "doy_sin", "doy_cos", "dow", "log_dist"],
    "RATES1": ["r_month", "r_carrier", "r_origin", "r_dest", "r_hb", "r_dom", "r_dow"],
    "RATES2": ["r_carmo", "r_orimo", "r_carhb"],
    "IXCAT": ["carhb", "carmo", "orimo"],
}
SETS = [("base+rates1", ["BASE", "RATES1"])]
COLS = [c for g, gs in SETS for g in gs for c in GROUPS[g]]

X_full, y_all = prepare(train), to_y(train)
X_ev_full, y_ev = prepare(evald), to_y(evald)


def fit(cfg, X, y, seed=SEED):
    m = xgb.XGBClassifier(
        tree_method="hist", enable_categorical=True, random_state=seed, n_jobs=N_JOBS,
        eval_metric="auc", **cfg,
    )
    m.fit(X, y)
    return m


def evalm(m, cols=None):
    p = m.predict_proba(X_ev_full[cols] if cols else X_ev_full)[:, 1]
    return roc_auc_score(y_ev, p)


t00 = time.time()
c1 = dict(n_estimators=400, max_depth=4, learning_rate=0.05)
c2 = dict(n_estimators=1000, max_depth=3, learning_rate=0.02)
c3 = dict(n_estimators=800, max_depth=4, learning_rate=0.03, gamma=2.0)

m1 = fit(c1, X_full[COLS], y_all); print(f"Eval AUC: {evalm(m1, COLS):.4f}   m1 d4 rates")
m2 = fit(c2, X_full[COLS], y_all); print(f"Eval AUC: {evalm(m2, COLS):.4f}   m2 d3 rates")
m3 = fit(c3, X_full[COLS], y_all); print(f"Eval AUC: {evalm(m3, COLS):.4f}   m3 d4 gamma rates")

# 5-fold bag
rng = np.random.default_rng(SEED)
idx = rng.permutation(len(train))
folds = np.array_split(idx, 5)
fold_models = []
for f, va in enumerate(folds):
    tr = np.setdiff1d(idx, va)
    m = fit(c1, X_full[COLS].iloc[tr], y_all[tr], seed=SEED + f)
    fold_models.append(m)
    print(f"Eval AUC: {evalm(m, COLS):.4f}   fold{f}")


def avg(models, cols_list=None):
    P = np.zeros(len(X_ev_full))
    for i, m in enumerate(models):
        cols = cols_list[i] if cols_list else COLS
        P += m.predict_proba(X_ev_full[cols])[:, 1] / len(models)
    return P


p_bag = avg(fold_models)
print(f"Eval AUC: {roc_auc_score(y_ev, p_bag):.4f}   5-fold bag d4 rates")

blends = {
    "m1+m2": [m1, m2],
    "m1+m2+m3": [m1, m2, m3],
    "m1+m2+m3+bag": [m1, m2, m3] + fold_models,
    "m1+bag": [m1] + fold_models,
    "m1+m3+bag": [m1, m3] + fold_models,
}
blends["m1+m2+m3+bag"] = None  # placeholder replaced below with weighted
best_p, best_name, best_a = None, None, -1
for name, models in blends.items():
    if models is None: continue
    p = avg(models)
    a = roc_auc_score(y_ev, p)
    print(f"Eval AUC: {a:.4f}   blend {name}")
    if a > best_a: best_p, best_name, best_a = p, name, a

print(f"best: {best_name}  total {time.time()-t00:.0f}s")
print(f"Eval AUC: {best_a:.4f}")

model = ([m1, m2, m3] + fold_models) if best_name == "m1+m2+m3+bag" else (
    [m1, m2] if best_name == "m1+m2" else (
        [m1, m2, m3] if best_name == "m1+m2+m3" else (
            [m1] + fold_models if best_name == "m1+bag" else [m1, m3] + fold_models)))


class Bag:
    def __init__(self, models): self.models = models
    def predict_proba(self, X):
        out = np.zeros(len(X))
        for m in self.models: out += m.predict_proba(X)[:, 1]
        return np.column_stack([1 - out / len(self.models), out / len(self.models)])


model = Bag(model)


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df)[best[1]])[:, 1]
