"""XGBoost binary classifier for airline dep_delayed_15min. Only file the agent edits.

Contract: `python train.py` -> prints `Eval AUC: 0.xxxx`; module-level predict_proba(df) -> P(positive).
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
ID_COLS = TASK.get("id_columns", [])
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}

# --- smoothed target encoding, fit on TRAIN only ---------------------------------
y_all = (train[TARGET] == POSITIVE).astype(int).to_numpy()
GLOBAL_MEAN = float(y_all.mean())
TE_SMOOTH = {"Origin": 50.0, "Dest": 50.0, "UniqueCarrier": 20.0, "Route": 100.0, "HourCat": 30.0}
NFOLD = 5


def _hourcat(df: pd.DataFrame) -> pd.Series:
    return np.floor(df["DepTime"].astype("float64") / 100.0).astype(int).astype(str)


_te_source = {
    "Origin": train["Origin"],
    "Dest": train["Dest"],
    "UniqueCarrier": train["UniqueCarrier"],
    "Route": train["Origin"].astype(str) + "_" + train["Dest"].astype(str),
    "HourCat": _hourcat(train),
}
te_maps = {}
oof_te = {}
_kf = KFold(n_splits=NFOLD, shuffle=True, random_state=SEED)
_folds = list(_kf.split(np.zeros(len(train))))
for col, src in _te_source.items():
    m = TE_SMOOTH[col]
    src = src.astype(str).to_numpy()
    stats = pd.DataFrame({"k": src, "y": y_all}).groupby("k")["y"].agg(["sum", "count"])
    te_maps[col] = ((stats["sum"] + m * GLOBAL_MEAN) / (stats["count"] + m)).to_dict()
    oof = np.zeros(len(train))
    for tr_idx, va_idx in _folds:
        st = pd.DataFrame({"k": src[tr_idx], "y": y_all[tr_idx]}).groupby("k")["y"].agg(["sum", "count"])
        fold_map = ((st["sum"] + m * GLOBAL_MEAN) / (st["count"] + m)).to_dict()
        oof[va_idx] = pd.Series(src[va_idx]).map(fold_map).to_numpy()
    oof_te[col] = oof


def prepare(df: pd.DataFrame, variant: int = 0) -> pd.DataFrame:
    # ALL feature engineering lives here (called on unseen rows by predict_proba).
    X = pd.DataFrame(index=df.index)
    dt = df["DepTime"].astype("float64")
    X["DepTime"] = dt
    X["Distance"] = df["Distance"].astype("float64")
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    if variant == 1:
        hour = np.floor(dt / 100.0)
        X["Hour"] = hour
        X["Minute"] = dt - hour * 100.0
        tmod = ((hour * 60.0 + (dt - hour * 100.0)) % 1440.0) / 1440.0
        X["TimeSin"] = np.sin(2 * np.pi * tmod)
        X["TimeCos"] = np.cos(2 * np.pi * tmod)
        X["LateNight"] = (dt >= 2400).astype("float64")
        X["LogDist"] = np.log1p(X["Distance"])
    elif variant == 2:
        X["Route"] = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
        X["HourCat"] = _hourcat(df)
        for col in TE_SMOOTH:
            X[f"TE_{col}"] = X[col].astype(str).map(te_maps[col]).astype("float64")
        X = X.drop(columns=["Route", "HourCat"])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


rng = np.random.RandomState(0)
CONFIGS = []
for i in range(50):
    CONFIGS.append(
        dict(
            n_estimators=int(rng.choice([25, 30, 40])),
            max_depth=int(rng.choice([4, 5, 6, 6, 7, 8])),
            learning_rate=float(rng.choice([0.1, 0.1, 0.15])),
            subsample=float(rng.choice([0.7, 0.8, 0.9])),
            colsample_bytree=float(rng.choice([0.6, 0.7, 0.8, 0.9, 1.0])),
            min_child_weight=float(rng.choice([1.0, 5.0, 10.0])),
            reg_lambda=float(rng.choice([1.0, 1.0, 5.0])),
            gamma=float(rng.choice([0.0, 0.1])),
        )
    )
VARIANTS = [int(v) for v in rng.randint(0, 3, size=50)]


def make_model(seed: int, cfg: dict) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        **cfg,
        tree_method="hist",
        enable_categorical=True,
        random_state=seed,
        n_jobs=N_JOBS,
    )


t0 = time.time()
X_variants = {v: prepare(train, v) for v in (0, 1, 2)}
X_variants[2] = X_variants[2].copy()
for col in TE_SMOOTH:  # leak-free OOF encodings for fitting
    X_variants[2][f"TE_{col}"] = oof_te[col]
y_train = to_y(train)
N_MEMBERS = len(CONFIGS)

# --- OOF predictions for greedy ensemble selection ---------------------------------
oof_preds = np.zeros((len(train), N_MEMBERS))
for f, (tr_idx, va_idx) in enumerate(_folds):
    for j in range(N_MEMBERS):
        cfg, v = CONFIGS[j], VARIANTS[j]
        m = make_model(f * 1000 + j, cfg).fit(X_variants[v].iloc[tr_idx], y_train[tr_idx])
        oof_preds[va_idx, j] = m.predict_proba(X_variants[v].iloc[va_idx])[:, 1]
print(f"OOF time: {time.time() - t0:.1f}s")

# greedy selection with replacement, on a subsample, with rank-based AUC
t1 = time.time()
sub = np.arange(0, len(train), 2)
y_sub = y_train[sub]
n_pos = int(y_sub.sum())
n_neg = len(y_sub) - n_pos


def fast_auc(p: np.ndarray) -> float:
    r = np.argsort(np.argsort(p, kind="stable"), kind="stable").astype("float64")
    return (r[y_sub == 1].sum() - n_pos * (n_pos - 1) / 2.0) / (n_pos * n_neg)


oof_sub = oof_preds[sub]
order = rng.permutation(N_MEMBERS)
bag = np.zeros(N_MEMBERS, dtype=int)
bag_sum = 0
bag_pred = np.zeros(len(sub))
best_auc = 0.5
for step in range(40):
    cand_best, cand_j = best_auc, -1
    for j in order:
        trial = (bag_pred * bag_sum + oof_sub[:, j]) / (bag_sum + 1)
        a = fast_auc(trial)
        if a > cand_best:
            cand_best, cand_j = a, j
    if cand_j < 0:
        break
    bag[cand_j] += 1
    bag_sum += 1
    bag_pred = (bag_pred * (bag_sum - 1) + oof_sub[:, cand_j]) / bag_sum
    best_auc = cand_best
print(f"Greedy OOF AUC: {best_auc:.4f} members={bag_sum} time={time.time() - t1:.1f}s")

sel = [(j, c) for j, c in enumerate(bag) if c > 0]
models = []
for j, _ in sel:
    v = VARIANTS[j]
    Xv = X_variants[2] if v == 2 else X_variants[v]
    models.append((make_model(j + 42, CONFIGS[j]).fit(Xv, y_train), v, CONFIGS[j]))
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    preds = []
    X_cache: dict = {}
    for (m, v, cfg), (j, c) in zip(models, sel):
        if v not in X_cache:
            X_cache[v] = prepare(df, v)
        preds.extend([m.predict_proba(X_cache[v])[:, 1]] * c)
    return np.mean(preds, axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
