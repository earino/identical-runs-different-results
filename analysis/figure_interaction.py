#!/usr/bin/env python3
"""Agent x model interaction figures, one per study, compliant runs only.

Each figure shows the same six means two ways:
  left   agents on the axis, one line per model: how far apart the agents are on each model
  right  models on the axis, one line per agent: what the model change does for each agent
Every point carries two intervals of different kinds, and the legend says so: a wide pale band where 80% of single
runs land (10th to 90th percentile, the statistic figure 1 marks in red) and the 95% interval of the mean (1.96 SE,
as the paper's tables compute it). The mean is pinned down tightly; a single run can land anywhere on the band.
Colour is the agent and marker shape the model (scripts/briefs/identity.py); lines that join a model's points are
neutral, dashed for GLM-5.3 Flash. Both figures share one y range, so the two sections compare directly.

Also prints, per study, every number the paper quotes from these figures: each agent's change with its SE, pi's change
minus OpenCode's, the spread of agents on each model with pi minus OpenCode, and whether each mean lies inside every
pairing's 10th-90th percentile band. SEs of differences use the normal approximation (square root of the summed
squared SEs), the method behind the paper's other SE figures.

Usage (repo root): python experiments/run_variance/figure_interaction.py [STUDY2_CELLS STUDY3_CELLS [OUT_DIR]]
  defaults: results/variance/cells.csv and results/variance_glm53/cells.csv, writing each study's figure to
  results/variance{,_glm53}/figures/interaction.{png,pdf}; with OUT_DIR, both go there as interaction_study{2,3}.*
"""
import csv, os, statistics as st, sys
from collections import defaultdict
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "briefs"))   # in the data repository, identity.py sits beside this file
from identity import AGENT_NAME, MODEL_MARKER, MODEL_NAME, NEUTRAL, agent_colour, marker_kw  # noqa: E402

rcParams.update({"font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "font.size": 11,
                 "axes.edgecolor": "#98A2AE", "axes.linewidth": 0.8, "axes.labelcolor": "#465061",
                 "xtick.color": "#465061", "ytick.color": "#465061", "text.color": "#1B2430",
                 "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150, "savefig.dpi": 300,
                 "pdf.fonttype": 42})
GRID, INK, MUTED = "#D6DBE1", "#1B2430", "#465061"
AGENTS = ["pi", "hermes", "opencode"]
# neutral line for each model's join: dashed and light for the shared starting model, solid and dark for the other
MODEL_LINE = {"glm-5.3-flash": (NEUTRAL, (0, (4, 2))), "deepseek-4.1-flash": ("#56606E", "-"), "glm-5.3": ("#56606E", "-")}


def load(path):
    out, base = defaultdict(list), None
    for r in csv.DictReader(open(path)):
        if r["counted"] != "True" or r["status"] != "scored" or not r["holdout_auc"]:
            continue
        if r["compliant"] != "True":
            continue
        out[(r["harness"], r["model"])].append(float(r["holdout_auc"]))
        base = base or float(r["baseline_auc"])
    return out, base


S2_CELLS = sys.argv[1] if len(sys.argv) > 1 else "results/variance/cells.csv"
S3_CELLS = sys.argv[2] if len(sys.argv) > 2 else "results/variance_glm53/cells.csv"
OUT = Path(sys.argv[3]) if len(sys.argv) > 3 else None
data, base = load(S2_CELLS)
data.update(load(S3_CELLS)[0])
STUDIES = [("Study 2", ["glm-5.3-flash", "deepseek-4.1-flash"], "the model change",
            OUT / "interaction_study2" if OUT else Path("results/variance/figures/interaction")),
           ("Study 3", ["glm-5.3-flash", "glm-5.3"], "the larger model",
            OUT / "interaction_study3" if OUT else Path("results/variance_glm53/figures/interaction"))]


def stats(v):
    q10, q90 = np.quantile(v, [0.1, 0.9])
    return st.mean(v), 1.96 * st.stdev(v) / len(v) ** 0.5, q10, q90


used = [data[(h, m)] for _, models, _, _ in STUDIES for h in AGENTS for m in models]
YLIM = (min(min(stats(v)[2] for v in used), base) - 0.002, max(stats(v)[3] for v in used) + 0.002)


def point(ax, x, h, m):
    mean, ci, q10, q90 = stats(data[(h, m)])
    c = agent_colour(h)
    ax.plot([x, x], [q10, q90], color=c, alpha=0.26, lw=7, solid_capstyle="butt", zorder=1)
    ax.errorbar([x], [mean], yerr=[ci], fmt="none", ecolor=c, elinewidth=1.8, capsize=3, capthick=1.4, zorder=3)
    ax.scatter([x], [mean], zorder=4, **marker_kw(m, c, 44, edge=1.0))
    return mean


def end_labels(ax, labels, x, gap):
    labels.sort()                                  # keep their order, push apart any that would overlap
    for i in range(1, len(labels)):
        labels[i][0] = max(labels[i][0], labels[i - 1][0] + gap)
    for y, txt in labels:
        ax.text(x, y, txt, va="center", fontsize=9, color=INK, linespacing=1.25)


def agents_on_x(ax, models):
    labels = []
    for j, m in enumerate(models):
        xs = [i + (j - 0.5) * 0.18 for i in range(len(AGENTS))]
        means = [point(ax, x, h, m) for x, h in zip(xs, AGENTS)]
        col, ls = MODEL_LINE[m]
        ax.plot(xs, means, color=col, lw=1.8, ls=ls, zorder=2)
        labels.append([means[-1], f"{MODEL_NAME[m]}\nspread {max(means) - min(means):.4f}"])
    end_labels(ax, labels, len(AGENTS) - 1 + 0.28, gap=0.0095)
    ax.set_xticks(range(len(AGENTS))); ax.set_xticklabels([AGENT_NAME[h] for h in AGENTS], fontsize=9.5)
    ax.set_xlim(-0.4, len(AGENTS) - 1 + 1.3)


def models_on_x(ax, models):
    labels = []
    for k, h in enumerate(AGENTS):
        xs = [i + (k - 1) * 0.11 for i in range(len(models))]
        means = [point(ax, x, h, m) for x, m in zip(xs, models)]
        ax.plot(xs, means, color=agent_colour(h), lw=2, zorder=2)
        labels.append([means[-1], f"{AGENT_NAME[h]}  {means[-1] - means[0]:+.4f}"])
    end_labels(ax, labels, len(models) - 1 + 0.26, gap=0.0028)
    ax.set_xticks(range(len(models))); ax.set_xticklabels([MODEL_NAME[m] for m in models], fontsize=9.5)
    ax.set_xlim(-0.35, len(models) - 1 + 0.95)


def finish(ax, title):
    ax.set_ylim(*YLIM)
    ax.axhline(base, color=MUTED, lw=0.9, ls=(0, (1, 2)), zorder=0)
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True); ax.tick_params(labelsize=8.5)
    ax.set_title(title, fontsize=10, color=INK, loc="left", pad=8)


def report(study, models):
    se = lambda h, m: stats(data[(h, m)])[1] / 1.96
    diff = lambda a, b: (a[0] - b[0], (a[1] ** 2 + b[1] ** 2) ** 0.5)          # (difference, its SE)
    lo_m, hi_m = models
    print(f"{study}: {MODEL_NAME[lo_m]} to {MODEL_NAME[hi_m]}, compliant runs")
    change = {}
    for h in AGENTS:
        change[h] = diff((stats(data[(h, hi_m)])[0], se(h, hi_m)), (stats(data[(h, lo_m)])[0], se(h, lo_m)))
        print(f"  {AGENT_NAME[h]:9s} {stats(data[(h, lo_m)])[0]:.4f} -> {stats(data[(h, hi_m)])[0]:.4f}   change "
              f"{change[h][0]:+.4f}  (SE {change[h][1]:.4f}, {change[h][0] / change[h][1]:.1f} SE)")
    d, sd = diff(change["pi"], change["opencode"])
    print(f"  pi's change minus OpenCode's: {d:+.4f}  ({d / sd:.1f} SE)")
    for m in models:
        ms = [stats(data[(h, m)])[0] for h in AGENTS]
        d, sd = diff((stats(data[("pi", m)])[0], se("pi", m)), (stats(data[("opencode", m)])[0], se("opencode", m)))
        print(f"  on {MODEL_NAME[m]:18s} agents spread {max(ms) - min(ms):.4f}; pi minus OpenCode {d:+.4f} ({d / sd:+.1f} SE)")
    keys = [(h, m) for h in AGENTS for m in models]
    inside = all(stats(data[b])[2] <= stats(data[a])[0] <= stats(data[b])[3] for a in keys for b in keys)
    print(f"  every mean inside every pairing's 10th-90th percentile band: {'yes' if inside else 'no'}")


for study, models, change, out in STUDIES:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.4, float(os.environ.get("INTERACTION_HEIGHT", 4.3))), sharey=True)
    agents_on_x(a, models); finish(a, "How far apart the agents are on each model")
    models_on_x(b, models); finish(b, f"What {change} does for each agent")
    a.set_ylabel("holdout AUC, compliant runs", fontsize=9.5)
    shapes = [Line2D([], [], ls="", color=NEUTRAL, marker=MODEL_MARKER[m][0], markersize=6, label=MODEL_NAME[m],
                     markerfacecolor="white" if MODEL_MARKER[m][1] else NEUTRAL, markeredgewidth=1.3)
              for m in models]
    handles = [Line2D([], [], color=NEUTRAL, alpha=0.4, lw=7, label="where 80% of single runs land"),
               *shapes,
               Line2D([], [], color=INK, lw=1.8, marker="o", markersize=5.5, markeredgecolor="white",
                      label="mean and its 95% interval"),
               Line2D([], [], color=MUTED, lw=0.9, ls=(0, (1, 2)), label=f"the starting code ({base:.4f})")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.01),
               handlelength=1.6, columnspacing=1.3, handletextpad=0.8)
    fig.suptitle(f"{study}: the same means two ways (colour: agent; shape: model; compliant runs)",
                 fontsize=11, color=INK, x=0.015, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight"); fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    report(study, models)
    print(f"  wrote {out}.png|pdf")
