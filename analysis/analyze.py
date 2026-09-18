#!/usr/bin/env python3
"""Run-variance experiment: per-run table and per-pair distribution of holdout AUC.

Reads every pulled box under PULLS (default pulls/variance, one dir per box as written by pull.sh) and writes to OUT
(default results/variance): cells.csv (one row per run, EVERY run: unscored, holdout_error, provider_error and
set-aside originals included and marked), summary.csv (per harness x model), boxes.csv (per pair x box means), and a
printed table. Rules (README): agent-caused outcomes count and are reported as rates next to the score distribution;
provider_error runs are marked and their clean re-run is the counted one; refit/violation flags are marked, never
dropped; stats are shown for all scored runs AND for clean runs side by side.
Never reads pulls/phase2_* or writes results/merged / results/phase2: this experiment is kept separate by design.

Usage (from the main checkout): python <worktree>/experiments/run_variance/analyze.py [PULLS] [OUT]
"""
import csv, glob, json, math, os, statistics as st, sys
from collections import defaultdict
from pathlib import Path

# The provider-error flag is recomputed from the logs with this checkout's detector: the image on the boxes matched `"429`
# inside commit hashes, and the loop's final best-commit re-evaluation writes that old flag back into eval.json.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench.evaluate import provider_error as detect_provider_error  # noqa: E402
from audit_frame_stats import audit_run  # noqa: E402  (contract rule 2: statistics on the frame being scored)
from audit_eval_training import eval_trained, delivered  # noqa: E402  (evaluation labels in the delivered code)

PULLS = sys.argv[1] if len(sys.argv) > 1 else "pulls/variance"
OUT = sys.argv[2] if len(sys.argv) > 2 else "results/variance"
REFIT_GAP = 0.03                                   # same screen as scripts/merge_report.py


def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def row_for(run_dir, box, counted, source):
    e = load(os.path.join(run_dir, "eval.json")) or {}
    m = load(os.path.join(run_dir, "meta.json")) or {}
    parts = run_dir.rstrip("/").split(os.sep)
    harness, model, seed_s = parts[-3], parts[-2], parts[-1]
    seed = int(seed_s.replace("seed", "").split("_")[0])
    h, be = e.get("holdout_auc"), e.get("best_eval_auc")
    logs = Path(run_dir) / "logs"
    # interim pulls carry only eval.json/meta.json: with no logs to scan, fall back to the flag the box recorded
    pe = (detect_provider_error(logs) if (logs / "harness.stdout").exists() else e.get("provider_error")) or (
        f"endpoint refusing requests at cell end: {m['endpoint_at_end']}" if m.get("endpoint_at_end") not in (None, "ok") else None)
    if not e:
        status = "unscored"                        # orchestration never produced eval.json (not an agent outcome)
    elif pe:
        status = "provider_error"
    elif h is None:
        status = "holdout_error"                   # agent delivered code that could not be scored: an agent outcome
    else:
        status = "scored"
    u = e.get("usage") or {}
    bch = e.get("best_commit_holdout_auc")
    fs_hits, _ = audit_run(run_dir)
    # audit_frame_stats is a screen; every hit was read. These two are false positives and do not count:
    #   opencode/deepseek-4.1-flash seed4  -> _oof_te(Xbase, y) is out-of-fold target encoding fitted on the
    #                                         training frame; it needs labels, so it never runs on a scored frame.
    #   hermes/deepseek-4.1-flash seed29   -> groupby only partitions rows; the values come from _SORTED_DEP,
    #                                         a table built from train.
    if (harness, model, seed) in {("opencode", "deepseek-4.1-flash", 4), ("hermes", "deepseek-4.1-flash", 29)}:
        fs_hits = []
    # Evaluation-set training is decided from the delivered train.py (audit_eval_training.py, hits read by a person), not
    # from refit_suspect: that screen compares the best experiment's evaluation score with the delivered code's holdout
    # score, so it flagged runs whose undelivered experiments trained on the evaluation rows and missed two that did.
    et, et_hits = eval_trained(harness, model, seed, delivered(run_dir))
    return {
        "box": box, "harness": harness, "model": model, "seed": seed, "counted": counted, "source": source, "status": status,
        "holdout_auc": h, "holdout_ap": e.get("holdout_ap"), "baseline_auc": e.get("baseline_holdout_auc"),
        "best_eval_auc": be, "eval_minus_holdout": round(be - h, 4) if (h is not None and be is not None) else None,
        "refit_suspect": bool(h is not None and be is not None and be - h > REFIT_GAP),
        "violations": ";".join(e.get("violations") or []),
        # substring hits in added train.py lines (docstrings count; e.g. a design note saying "eval/holdout are 2006") need a
        # human read; everything else (modified/missing protected files, data hash, holdout path in logs, installs) is hard
        "violations_hard": ";".join(v for v in (e.get("violations") or []) if not v.startswith("suspicious-train-py:")),
        "violations_review": ";".join(v for v in (e.get("violations") or []) if v.startswith("suspicious-train-py:")),
        # contract rule 2 ("never on the dataframe passed in"): traced from predict_proba, not grepped. A screen: read the lines.
        "frame_stats": bool(fs_hits), "frame_stats_kinds": ";".join(sorted({h[1] for h in fs_hits})),
        "frame_stats_lines": " | ".join(f"{h[0]}: {h[2]}" for h in fs_hits[:3]),
        "best_commit_holdout_auc": bch, "delivered_minus_best_commit": round(h - bch, 4) if (h is not None and bch is not None) else None,
        "n_experiments": e.get("n_experiments"), "n_ok": e.get("n_ok"), "n_timeout": e.get("n_timeout"), "n_crash": e.get("n_crash"),
        "timed_out": e.get("timed_out", m.get("timed_out")), "stalled": e.get("stalled", m.get("stalled")),
        "disk_blowup": m.get("disk_blowup"), "cpu_killed": e.get("cpu_killed", m.get("cpu_killed")),
        "cpu_refusals": e.get("cpu_refusals"), "cpu_over_budget": e.get("cpu_over_budget"),
        "fits_total": e.get("fits_total"), "fits_uncounted": e.get("fits_uncounted"),
        "cpu_seconds_counted": e.get("cpu_seconds_counted"), "cpu_seconds_uncounted": e.get("cpu_seconds_uncounted"),
        "cpu_seconds_container": e.get("cpu_seconds_container", m.get("cpu_seconds_container")),
        "exit_code": m.get("exit_code"), "endpoint_at_end": m.get("endpoint_at_end"), "final_md_written": e.get("final_md_written"),
        "provider_error": (pe or "")[:80], "provider_error_recorded": (e.get("provider_error") or "")[:80],
        "holdout_error": ((e.get("holdout_error") or "").strip().splitlines() or [""])[-1][:90],
        "wall_minutes": round(m["wall_seconds"] / 60, 1) if m.get("wall_seconds") else None,
        "tokens_in": u.get("tokens_in"), "tokens_out": u.get("tokens_out"),
        "started": m.get("started"), "ended": m.get("ended"), "cpuset": (m.get("container") or {}).get("cpuset"),
        "scored_on": e.get("scored_on"),
        # the compliance record every figure and table uses: evaluation labels in the delivered code, statistics from the
        # scored frame. refit_suspect above stays as the screen's raw flag, for the record.
        "eval_trained": et, "eval_trained_evidence": (et_hits[0] if et and et_hits else "")[:100],
        "compliant": status == "scored" and not et and not fs_hits,
    }


rows = []
for box_dir in sorted(glob.glob(os.path.join(PULLS, "*"))):
    box = os.path.basename(box_dir)
    for d in sorted(glob.glob(os.path.join(box_dir, "runs", "airline", "*", "*", "seed*"))):
        rows.append(row_for(d, box, True, "runs"))
    for d in sorted(glob.glob(os.path.join(box_dir, "runs_smoke", "variance_flagged", "airline", "*", "*", "seed*"))):
        rows.append(row_for(d, box, False, "set_aside"))
if not rows:
    sys.exit(f"no runs under {PULLS}/<box>/runs/airline")
rows.sort(key=lambda r: (r["harness"], r["model"], r["seed"], not r["counted"]))
os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "cells.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


def dist(vals):
    if not vals:
        return {}
    v = sorted(vals); n = len(v)
    q = st.quantiles(v, n=20, method="inclusive") if n >= 2 else [v[0]] * 19   # 5% steps
    return {"n": n, "mean": st.mean(v), "sd": st.stdev(v) if n >= 2 else None, "min": v[0], "p05": q[0], "p25": q[4],
            "median": st.median(v), "p75": q[14], "p95": q[18], "max": v[-1], "range": v[-1] - v[0], "iqr": q[14] - q[4]}


summary, boxes = [], []
for (hn, mn) in sorted({(r["harness"], r["model"]) for r in rows}):
    R = [r for r in rows if r["harness"] == hn and r["model"] == mn and r["counted"]]
    scored = [r for r in R if r["status"] == "scored"]
    # "clean" = compliant: no evaluation labels in the delivered code's training, no statistics from the scored frame (the
    # paper's two data rules). Protocol breaches that touch neither, such as committing a notes file, are listed in
    # violations_hard and do not make a run noncompliant. Flagged runs are never dropped from `runs`/`all_*`.
    clean = [r for r in scored if r["compliant"]]
    s = {"harness": hn, "model": mn, "runs": len(R),
         "scored": len(scored), "holdout_error": sum(r["status"] == "holdout_error" for r in R),
         "unscored": sum(r["status"] == "unscored" for r in R), "provider_error": sum(r["status"] == "provider_error" for r in R),
         "set_aside_originals": sum(1 for r in rows if r["harness"] == hn and r["model"] == mn and not r["counted"]),
         "zero_experiments": sum((r["n_experiments"] or 0) == 0 for r in scored),
         "timed_out": sum(bool(r["timed_out"]) for r in R), "stalled": sum(bool(r["stalled"]) for r in R),
         "cpu_killed": sum(bool(r["cpu_killed"]) for r in R), "refit_suspect": sum(r["refit_suspect"] for r in scored), "eval_trained": sum(r["eval_trained"] for r in scored),
         "violations_hard": sum(bool(r["violations_hard"]) for r in scored), "violations_review": sum(bool(r["violations_review"]) for r in scored),
         "frame_stats": sum(bool(r["frame_stats"]) for r in scored), "fits_uncounted_runs": sum((r["fits_uncounted"] or 0) > 0 for r in scored)}
    for tag, sub in (("all", scored), ("clean", clean)):
        for k, v in dist([r["holdout_auc"] for r in sub]).items():
            s[f"{tag}_{k}"] = round(v, 5) if isinstance(v, float) else v
    summary.append(s)
    # box as a factor: per-box mean and a one-way ANOVA F on scored runs (all pairs run on all boxes by design)
    by_box = defaultdict(list)
    for r in scored:
        by_box[r["box"]].append(r["holdout_auc"])
    groups = [g for g in by_box.values() if g]
    F = None
    if len(groups) >= 2 and sum(len(g) for g in groups) > len(groups):
        allv = [x for g in groups for x in g]; gm = st.mean(allv)
        ssb = sum(len(g) * (st.mean(g) - gm) ** 2 for g in groups); ssw = sum((x - st.mean(g)) ** 2 for g in groups for x in g)
        dfb, dfw = len(groups) - 1, len(allv) - len(groups)
        F = round((ssb / dfb) / (ssw / dfw), 3) if ssw > 0 else math.inf
    for b in sorted(by_box):
        boxes.append({"harness": hn, "model": mn, "box": b, "n": len(by_box[b]), "mean": round(st.mean(by_box[b]), 5),
                      "sd": round(st.stdev(by_box[b]), 5) if len(by_box[b]) >= 2 else None, "pair_box_anova_F": F})

for name, data in (("summary.csv", summary), ("boxes.csv", boxes)):
    if data:
        keys = list(dict.fromkeys(k for d in data for k in d))
        with open(os.path.join(OUT, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(data)

f4 = lambda x: f"{x:.4f}" if isinstance(x, (int, float)) and x is not None else "  -   "
print("h_err = agent delivered unscorable code (counts as an outcome); unsc = no eval.json (orchestration, not the agent)")
print(f"{len(rows)} rows ({sum(r['counted'] for r in rows)} counted runs, {sum(not r['counted'] for r in rows)} set-aside originals) "
      f"from {PULLS} -> {OUT}/cells.csv, summary.csv, boxes.csv\n")
print(f"{'harness':9} {'model':19} {'runs':>4} {'scored':>6} {'h_err':>5} {'unsc':>4} {'mean':>7} {'sd':>7} {'p05':>7} {'median':>7} {'p95':>7} "
      f"{'range':>7} | {'clean n':>7} {'sd':>7} | flags")
for s in summary:
    flags = ", ".join(f"{k}={s[k]}" for k in ("provider_error", "eval_trained", "refit_suspect", "frame_stats", "violations_hard", "violations_review", "timed_out", "stalled", "cpu_killed", "zero_experiments") if s[k])
    print(f"{s['harness']:9} {s['model']:19} {s['runs']:4} {s['scored']:6} {s['holdout_error']:5} {s['unscored']:4} {f4(s.get('all_mean'))} {f4(s.get('all_sd'))} "
          f"{f4(s.get('all_p05'))} {f4(s.get('all_median'))} {f4(s.get('all_p95'))} {f4(s.get('all_range'))} | "
          f"{s.get('clean_n', 0):7} {f4(s.get('clean_sd'))} | {flags}")
