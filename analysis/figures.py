#!/usr/bin/env python3
"""Run-variance figures from results/variance/cells.csv (analyze.py output).

fig1 spread : one row per (harness, model), every scored run as a dot on the holdout-AUC axis, min-max rule, mean
              marker, SD labelled at the right. Rows sorted by SD: the question is how much identical runs differ.
fig2 hist   : the same six distributions as small-multiple histograms (bin 0.005), shared x.
Runs that trained on the labelled eval set (refit_suspect) are a second series, not hidden: they are part of the
distribution and they are what makes one pair's spread wide.

Usage (from the main checkout): python <worktree>/experiments/run_variance/figures.py [CELLS_CSV] [OUT_DIR]
"""
import csv, os, statistics as st, sys
from collections import defaultdict
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D

CELLS = Path(sys.argv[1] if len(sys.argv) > 1 else "results/variance/cells.csv")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "results/variance/figures")
OUT.mkdir(parents=True, exist_ok=True)

# repo figure style (scripts/briefs/make_figures.py) + validated categorical slots 1-2 (dataviz reference palette)
rcParams.update({"font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "font.size": 11,
                 "axes.edgecolor": "#98A2AE", "axes.linewidth": 0.8, "axes.labelcolor": "#465061",
                 "xtick.color": "#465061", "ytick.color": "#465061", "text.color": "#1B2430",
                 "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150, "savefig.dpi": 300,
                 "pdf.fonttype": 42})
BLUE, ORANGE, VIOLET, GRID, INK, MUTED = "#2a78d6", "#eb6834", "#4a3aa7", "#D6DBE1", "#1B2430", "#465061"

rows = [r for r in csv.DictReader(open(CELLS)) if r["counted"] == "True" and r["status"] == "scored" and r["holdout_auc"]]
pairs = defaultdict(list)
for r in rows:
    # 0 = played it straight, 1 = trained on the labelled eval set, 2 = statistics computed on the frame being scored
    kind = 1 if r["refit_suspect"] == "True" else (2 if r.get("frame_stats") == "True" else 0)
    pairs[(r["harness"], r["model"])].append((float(r["holdout_auc"]), kind))
baseline = next((float(r["baseline_auc"]) for r in rows if r["baseline_auc"]), None)
order = sorted(pairs, key=lambda k: st.stdev([v for v, _ in pairs[k]]) if len(pairs[k]) > 1 else 0)
# display names as the briefs write them, so a figure and its table agree
AGENT = {"pi": "pi", "opencode": "OpenCode", "hermes": "Hermes", "claude": "Claude Code", "codex": "Codex", "openclaw": "OpenClaw"}
MODEL = {"glm-5.3-flash": "GLM-5.3 Flash", "deepseek-4.1-flash": "DeepSeek 4.1 Flash", "glm-5.3": "GLM-5.3"}
label = lambda k: f"{AGENT.get(k[0], k[0])} / {MODEL.get(k[1], k[1])}"
lo = min(v for p in pairs.values() for v, _ in p) - 0.004
hi = max(v for p in pairs.values() for v, _ in p) + 0.004
ns = sorted({len(p) for p in pairs.values()})
runs_txt = f"{ns[0]} repeats" if len(ns) == 1 else f"{ns[0]}-{ns[-1]} repeats"   # partial data must not claim the full 52

# ---- fig1: every run as a dot, one row per pair
fig, ax = plt.subplots(figsize=(7.6, float(os.environ.get("VARIANCE_SPREAD_HEIGHT", 3.6))))
for i, k in enumerate(order):
    vals = [v for v, _ in pairs[k]]
    plain = [v for v, f in pairs[k] if f == 0]; pooled = [v for v, f in pairs[k] if f == 1]; framed = [v for v, f in pairs[k] if f == 2]
    ax.plot([min(vals), max(vals)], [i, i], color=GRID, lw=5, solid_capstyle="round", zorder=1)
    ax.scatter(plain, [i] * len(plain), s=16, color=BLUE, alpha=0.5, edgecolor="none", zorder=2)
    if pooled:
        ax.scatter(pooled, [i] * len(pooled), s=26, color=ORANGE, alpha=0.9, edgecolor="white", linewidth=0.6, zorder=4, marker="D")
    if framed:
        ax.scatter(framed, [i] * len(framed), s=30, color=VIOLET, alpha=0.9, edgecolor="white", linewidth=0.6, zorder=4, marker="^")
    ax.scatter([st.mean(vals)], [i], s=42, color=INK, edgecolor="white", linewidth=1.0, zorder=5)
    ax.text(hi + 0.002, i, f"SD {st.stdev(vals):.4f}   n={len(vals)}", va="center", fontsize=9, color=MUTED)
if baseline:
    # dotted, not dashed: fig2 uses a dash for the median, and a reader who meets that first reads this as a median too
    ax.axvline(baseline, color=MUTED, lw=0.9, ls=(0, (1, 2)), zorder=0)
ax.set_yticks(range(len(order))); ax.set_yticklabels([label(k) for k in order], fontsize=9.5)
ax.set_xlim(lo, hi); ax.set_ylim(-0.6, len(order) - 0.4); ax.set_xlabel("holdout AUC (one dot = one run)")
ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.set_title(f"Identical runs, different results: {runs_txt} of each agent + model", fontsize=11.5, color=INK, loc="left", pad=12)
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=BLUE, alpha=0.6, markersize=6, label="run"),
                   Line2D([], [], marker="D", ls="", color=ORANGE, markersize=6, label="trained on the labelled eval set"),
                   Line2D([], [], marker="^", ls="", color=VIOLET, markersize=6, label="statistics from the frame being scored"),
                   Line2D([], [], marker="o", ls="", color=INK, markersize=7, label="mean"),
                   Line2D([], [], color=MUTED, lw=0.9, ls=(0, (1, 2)), label=f"the starting code ({baseline:.4f})")],
          loc="lower right", bbox_to_anchor=(1.02, -0.46), ncol=5, frameon=False, fontsize=8.5, handletextpad=0.4, columnspacing=1.0)
fig.savefig(OUT / "spread.png", bbox_inches="tight"); fig.savefig(OUT / "spread.pdf", bbox_inches="tight"); plt.close(fig)

# ---- fig2: the same distributions as histograms
bw = 0.005
edges = [lo + i * bw for i in range(int((hi - lo) / bw) + 2)]
# VARIANCE_HIST_HEIGHT trims the height for the two-page brief (the default is the standalone figure)
fig, axes = plt.subplots(2, 3, figsize=(8.2, float(os.environ.get("VARIANCE_HIST_HEIGHT", 4.2))), sharex=True, sharey=True)
for ax, k in zip(axes.ravel(), order):
    plain = [v for v, f in pairs[k] if f == 0]; pooled = [v for v, f in pairs[k] if f == 1]; framed = [v for v, f in pairs[k] if f == 2]
    ax.hist([plain, pooled, framed], bins=edges, stacked=True, color=[BLUE, ORANGE, VIOLET], edgecolor="white", linewidth=0.5)
    vals = [v for v, _ in pairs[k]]
    ax.axvline(st.mean(vals), color=INK, lw=1.2)                      # mean: what a 3-run average estimates
    ax.axvline(st.median(vals), color=MUTED, lw=1.2, ls=(0, (3, 2)))  # median: where the flagged tails pull the mean away
    # two lines: the display names are long enough that one line collides with the next panel
    ax.set_title(f"{label(k)}\nSD {st.stdev([v for v, _ in pairs[k]]):.4f}", fontsize=8.5, color=INK, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True); ax.tick_params(labelsize=8.5)
for ax in axes[1]:
    ax.set_xlabel("holdout AUC", fontsize=9)
for ax in axes[:, 0]:
    ax.set_ylabel("runs", fontsize=9)
fig.legend(handles=[Line2D([], [], marker="s", ls="", color=BLUE, markersize=7, label="run"),
                           Line2D([], [], marker="s", ls="", color=ORANGE, markersize=7, label="trained on the labelled eval set"),
                           Line2D([], [], marker="s", ls="", color=VIOLET, markersize=7, label="statistics from the scored frame"),
                    Line2D([], [], color=INK, lw=1.2, label="mean"),
                    Line2D([], [], color=MUTED, lw=1.2, ls=(0, (3, 2)), label="median")],
           loc="lower center", ncol=5, frameon=False, fontsize=8.5, handletextpad=0.5, columnspacing=1.6, bbox_to_anchor=(0.5, -0.01))
fig.suptitle(f"Run-to-run distribution per agent + model ({runs_txt}, airline, LunaRoute)", fontsize=11, color=INK, x=0.02, ha="left")
fig.tight_layout(rect=(0, 0.05, 1, 0.95))   # room for the figure-level legend under the panels
fig.savefig(OUT / "histograms.png", bbox_inches="tight"); fig.savefig(OUT / "histograms.pdf", bbox_inches="tight"); plt.close(fig)

# ---- fig3: budget use against score, within each pairing, valid runs only
# The relationship is the paper's bridge between the two studies, so show it on the stronger data: one panel per
# pairing, so the reader sees it inside a fixed agent and model rather than pooled across them.
from scipy.stats import spearmanr  # noqa: E402

BUDGET = 18000.0
raw = [r for r in csv.DictReader(open(CELLS))
       if r["counted"] == "True" and r["status"] == "scored" and r["holdout_auc"]
       and r["refit_suspect"] != "True" and r["frame_stats"] != "True" and r["cpu_seconds_counted"]]
comp = defaultdict(list)
for r in raw:
    comp[(r["harness"], r["model"])].append(
        ((float(r["cpu_seconds_counted"] or 0) + float(r["cpu_seconds_uncounted"] or 0)) / BUDGET, float(r["holdout_auc"])))
order3 = sorted(comp, key=lambda k: -spearmanr(*zip(*comp[k])).statistic)
fig, axes = plt.subplots(2, 3, figsize=(8.2, float(os.environ.get("VARIANCE_COMPUTE_HEIGHT", 3.6))), sharex=True, sharey=True)
for ax, k in zip(axes.ravel(), order3):
    x, y = zip(*comp[k])
    ax.scatter(x, y, s=18, color=BLUE, alpha=0.55, edgecolor="none")
    if baseline:
        ax.axhline(baseline, color=MUTED, lw=0.9, ls=(0, (1, 2)))
    ax.set_title(f"{label(k)}\nrho {spearmanr(x, y).statistic:+.2f}, n={len(x)}", fontsize=8.5, color=INK, loc="left")
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True); ax.tick_params(labelsize=8.5)
for ax in axes[1]:
    ax.set_xlabel("share of the CPU budget used", fontsize=9)
for ax in axes[:, 0]:
    ax.set_ylabel("holdout AUC", fontsize=9)
fig.suptitle("Budget use against score inside each pairing (valid runs; dotted line: the starting code)",
             fontsize=11, color=INK, x=0.02, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(OUT / "compute.png", bbox_inches="tight"); fig.savefig(OUT / "compute.pdf", bbox_inches="tight"); plt.close(fig)
print(f"wrote {OUT}/spread.png|pdf, {OUT}/histograms.png|pdf and {OUT}/compute.png|pdf from {len(rows)} scored runs")
