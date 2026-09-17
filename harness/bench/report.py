"""Aggregate runs/**/eval.json into results/: long CSV, per-dataset harness x model grids, rankings, heatmaps."""
from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

from bench import ROOT

RESULTS = ROOT / "results"


def load_evals(runs_root: Path | None = None, include_provider_errors: bool = False) -> list[dict]:
    """Scored cells. Cells whose harness hit a provider quota/outage (eval.json `provider_error`) are excluded
    unless asked for: they did not get a fair run and `bench grid` re-runs them."""
    root = runs_root or ROOT / "runs"
    out = []
    for p in sorted(root.glob("*/*/*/seed*/eval.json")):
        try:
            e = json.loads(p.read_text())
        except Exception:
            continue
        if e.get("provider_error") and not include_provider_errors:
            continue
        out.append(e)
    return out


def _fmt(x, nd=4):
    return "" if x is None else f"{x:.{nd}f}"


def write_long_csv(evals: list[dict], path: Path) -> None:
    import csv
    cols = ["dataset", "harness", "model", "model_endpoint", "seed", "holdout_auc", "holdout_ap", "baseline_holdout_auc",
            "delta_vs_baseline", "best_eval_auc", "final_eval_auc", "generalization_gap", "n_experiments", "n_ok",
            "n_crash", "n_timeout", "n_commits", "wall_seconds", "timed_out", "exit_code", "final_md_written",
            "violations", "tokens_in", "tokens_in_cached", "cached_share", "tokens_out", "api_calls", "cost_usd", "holdout_error"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for e in evals:
            u = e.get("usage") or {}
            row = {**e, "tokens_in": u.get("tokens_in"), "tokens_in_cached": u.get("tokens_in_cached"), "cached_share": u.get("cached_share"),
                   "tokens_out": u.get("tokens_out"), "api_calls": u.get("api_calls"), "cost_usd": u.get("cost_usd"),
                   "violations": ";".join(e.get("violations") or [])}
            w.writerow([row.get(c, "") for c in cols])


def grid(evals: list[dict], dataset: str, metric: str = "holdout_auc"):
    cells = defaultdict(list)
    harnesses, models = [], []
    for e in evals:
        if e["dataset"] != dataset:
            continue
        if e["harness"] not in harnesses:
            harnesses.append(e["harness"])
        if e["model"] not in models:
            models.append(e["model"])
        if e.get(metric) is not None:
            cells[(e["harness"], e["model"])].append(e[metric])
    return harnesses, models, cells


def grid_markdown(evals: list[dict], dataset: str, metric: str = "holdout_auc", nd: int = 4) -> str:
    harnesses, models, cells = grid(evals, dataset, metric)
    base = next((e.get("baseline_holdout_auc") for e in evals if e["dataset"] == dataset and e.get("baseline_holdout_auc") is not None), None)
    lines = [f"### {dataset} — {metric} (mean ± sd over seeds, n)", ""]
    if base is not None and metric == "holdout_auc":
        lines += [f"Untouched baseline train.py on holdout: **{base:.4f}**", ""]
    lines.append("| harness \\ model | " + " | ".join(models) + " | harness mean |")
    lines.append("|---|" + "---|" * (len(models) + 1))
    for h in harnesses:
        row, means = [], []
        for m in models:
            v = cells.get((h, m), [])
            if not v:
                row.append("—")
                continue
            mu = st.mean(v)
            means.append(mu)
            sd = st.stdev(v) if len(v) > 1 else 0.0
            row.append(f"{mu:.{nd}f} ± {sd:.{nd}f} (n={len(v)})" if len(v) > 1 else f"{mu:.{nd}f} (n=1)")
        lines.append(f"| **{h}** | " + " | ".join(row) + f" | {_fmt(st.mean(means), nd) if means else '—'} |")
    col_means = []
    for m in models:
        vals = [st.mean(cells[(h, m)]) for h in harnesses if cells.get((h, m))]
        col_means.append(_fmt(st.mean(vals), nd) if vals else "—")
    lines.append("| *model mean* | " + " | ".join(col_means) + " | |")
    return "\n".join(lines) + "\n"


def rankings(evals: list[dict]) -> str:
    """Average rank of each harness (within each model x dataset) and each model (within each harness x dataset)."""
    by_hm = defaultdict(list)
    for e in evals:
        if e.get("holdout_auc") is not None:
            by_hm[(e["dataset"], e["harness"], e["model"])].append(e["holdout_auc"])
    means = {k: st.mean(v) for k, v in by_hm.items()}
    h_ranks, m_ranks = defaultdict(list), defaultdict(list)
    for (d, h, m) in means:
        peers = sorted([(means[(d, hh, mm)], hh) for (dd, hh, mm) in means if dd == d and mm == m], reverse=True)
        h_ranks[h].append([p[1] for p in peers].index(h) + 1)
        peers = sorted([(means[(d, hh, mm)], mm) for (dd, hh, mm) in means if dd == d and hh == h], reverse=True)
        m_ranks[m].append([p[1] for p in peers].index(m) + 1)
    out = ["### Harness ranking (mean rank across model x dataset cells; 1 = best)", "", "| harness | mean rank | cells |", "|---|---|---|"]
    for h, r in sorted(h_ranks.items(), key=lambda kv: st.mean(kv[1])):
        out.append(f"| {h} | {st.mean(r):.2f} | {len(r)} |")
    out += ["", "### Model ranking (mean rank across harness x dataset cells; 1 = best)", "", "| model | mean rank | cells |", "|---|---|---|"]
    for m, r in sorted(m_ranks.items(), key=lambda kv: st.mean(kv[1])):
        out.append(f"| {m} | {st.mean(r):.2f} | {len(r)} |")
    return "\n".join(out) + "\n"


def reliability(evals: list[dict]) -> str:
    rows = defaultdict(lambda: {"runs": 0, "scored": 0, "timed_out": 0, "violations": 0, "exp": [], "crash": 0, "final": 0})
    for e in evals:
        r = rows[e["harness"]]
        r["runs"] += 1
        r["scored"] += e.get("holdout_auc") is not None
        r["timed_out"] += bool(e.get("timed_out")) or bool(e.get("stalled")) or bool(e.get("disk_blowup"))
        r["violations"] += bool(e.get("violations"))
        r["exp"].append(e.get("n_experiments") or 0)
        r["crash"] += e.get("n_crash") or 0
        r["final"] += bool(e.get("final_md_written"))
    out = ["### Reliability per harness", "", "| harness | runs | scored | hit ceiling / stalled | protocol violations | mean experiments | crashes | wrote FINAL.md |", "|---|---|---|---|---|---|---|---|"]
    for h, r in sorted(rows.items()):
        out.append(f"| {h} | {r['runs']} | {r['scored']} | {r['timed_out']} | {r['violations']} | {st.mean(r['exp']):.1f} | {r['crash']} | {r['final']} |")
    return "\n".join(out) + "\n"


def economics(evals: list[dict], dataset: str) -> str:
    """Per harness x model: mean input tokens (M), cached share, cost per cell, and cost per +0.001 holdout AUC over baseline."""
    cells = defaultdict(list)
    harnesses, models = [], []
    for e in evals:
        if e["dataset"] != dataset:
            continue
        if e["harness"] not in harnesses:
            harnesses.append(e["harness"])
        if e["model"] not in models:
            models.append(e["model"])
        cells[(e["harness"], e["model"])].append(e)
    out = [f"### {dataset} — economics (mean per cell over seeds)", "",
           "| harness | model | input tokens (M) | cached share | output tokens (k) | cost / cell | Δ AUC vs baseline | $ per +0.001 AUC |",
           "|---|---|---|---|---|---|---|---|"]
    h_tot = defaultdict(lambda: [0.0, 0])
    for h in harnesses:
        for m in models:
            es = cells.get((h, m), [])
            us = [e["usage"] for e in es if (e.get("usage") or {}).get("tokens_in")]
            if not us:
                continue
            tin = st.mean(x["tokens_in"] for x in us) / 1e6
            cs = [x["cached_share"] for x in us if x.get("cached_share") is not None]
            tout = st.mean(x.get("tokens_out") or 0 for x in us) / 1e3
            costs = [x["cost_usd"] for x in us if x.get("cost_usd") is not None]
            cost = st.mean(costs) if costs else None
            deltas = [e["delta_vs_baseline"] for e in es if e.get("delta_vs_baseline") is not None]
            d = st.mean(deltas) if deltas else None
            per = (cost / (d / 0.001)) if cost is not None and d is not None and d >= 0.0005 else None   # below half a point: not meaningful
            if cost is not None:
                h_tot[h][0] += cost * len(costs); h_tot[h][1] += len(costs)
            out.append(f"| {h} | {m} | {tin:.2f} | {_fmt(st.mean(cs), 2) if cs else '—'} | {tout:.1f} | "
                       f"{'$' + _fmt(cost, 2) if cost is not None else '—'} | {_fmt(d, 4) if d is not None else '—'} | "
                       f"{'$' + _fmt(per, 2) if per is not None else '—'} |")
    out += ["", "| harness | total spend on this dataset | cells costed |", "|---|---|---|"]
    for h, (tot, n) in sorted(h_tot.items(), key=lambda kv: -kv[1][0]):
        out.append(f"| {h} | ${tot:.2f} | {n} |")
    out += ["", "Token counts are what each harness reports (Ollama calls them approximations); OpenClaw reports no cache split,",
            "so its cost assumes uncached input. Prices: `bench.yaml` `prices` (ollama.com/pricing, standard rate).", ""]
    return "\n".join(out)


def heatmap(evals: list[dict], dataset: str, path: Path, metric: str = "holdout_auc") -> None:
    """Harness x model heatmap. Sequential single-hue ramp (magnitude), values printed in ink, thin white cell gaps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap

    harnesses, models, cells = grid(evals, dataset, metric)
    if not harnesses or not models:
        return
    M = np.full((len(harnesses), len(models)), np.nan)
    for i, h in enumerate(harnesses):
        for j, m in enumerate(models):
            v = cells.get((h, m))
            if v:
                M[i, j] = st.mean(v)
    cmap = LinearSegmentedColormap.from_list("seq_blue", ["#e3ecf7", "#5b8fd6", "#1d4f91"])  # one hue, light -> dark
    cmap.set_bad("#f2f2f2")
    fig, ax = plt.subplots(figsize=(1.6 * len(models) + 3, 0.7 * len(harnesses) + 2))
    vmin, vmax = np.nanmin(M), np.nanmax(M)
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax if vmax > vmin else vmin + 1e-6, aspect="auto")
    ax.set_xticks(range(len(models)), models, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(harnesses)), harnesses, fontsize=9)
    ax.set_xticks(np.arange(-0.5, len(models), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(harnesses), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(len(harnesses)):
        for j in range(len(models)):
            if not np.isnan(M[i, j]):
                frac = (M[i, j] - vmin) / (vmax - vmin) if vmax > vmin else 0
                ax.text(j, i, f"{M[i, j]:.4f}", ha="center", va="center", fontsize=9,
                        color="#ffffff" if frac > 0.6 else "#1a1a1a")
    ax.set_title(f"{dataset}: mean {metric} (harness × model)", fontsize=11, loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def build_report(runs_root: Path | None = None) -> Path:
    RESULTS.mkdir(exist_ok=True)
    evals = load_evals(runs_root)
    n_provider = len(load_evals(runs_root, include_provider_errors=True)) - len(evals)
    write_long_csv(evals, RESULTS / "results.csv")
    datasets = sorted({e["dataset"] for e in evals})
    md = ["# Harness benchmark results", "", f"{len(evals)} scored runs across {len(datasets)} dataset(s)."
          + (f" {n_provider} cell(s) hit a provider quota/outage and are excluded pending re-run." if n_provider else ""), ""]
    for d in datasets:
        md.append(grid_markdown(evals, d, "holdout_auc"))
        md.append(grid_markdown(evals, d, "delta_vs_baseline"))
        heatmap(evals, d, RESULTS / f"heatmap_{d}.png")
        md.append(f"![{d} heatmap](heatmap_{d}.png)\n")
        md.append(economics(evals, d))
    if evals:
        md.append(rankings(evals))
        md.append(reliability(evals))
    (RESULTS / "README.md").write_text("\n".join(md))
    return RESULTS / "README.md"
