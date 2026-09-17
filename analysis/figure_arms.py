#!/usr/bin/env python3
"""Study 3 figure: GLM-5.3 against GLM-5.3 Flash, with the gain shown against the noise it has to clear.

Two panels, because the comparison has two halves and one of them is the point:
  left   every valid run of both models, three agents, so the reader sees the distributions overlap heavily
  right  the per-agent gain with its 95% interval, against a marked line at one run-to-run SD

The right panel is the argument. A reader who only sees means will read "+0.0097" as a clean win; seeing it land on
the one-SD line says the same number means you cannot detect it in a single run.

Colours: the two models are a validated categorical pair (blue #1a6fc4, green #2fa36b: CVD dE 22.6 protan / 9.2
tritan, normal-vision 23.4). Orange and violet are deliberately NOT used -- in this paper they already mean "this
run broke a task rule" (figures 10 and 11), and reusing them for a model would mislead. Marker shape carries the
model too, so identity never rests on colour alone.

Usage: python experiments/run_variance/figure_arms.py [PRO_CELLS] [FLASH_CELLS] [OUT_DIR]
"""
import csv, random, statistics as st, sys
from collections import defaultdict
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D

PRO = Path(sys.argv[1] if len(sys.argv) > 1 else "results/variance_glm53/cells.csv")
FLASH = Path(sys.argv[2] if len(sys.argv) > 2 else "results/variance/cells.csv")
OUT = Path(sys.argv[3] if len(sys.argv) > 3 else "results/variance_glm53/figures")
OUT.mkdir(parents=True, exist_ok=True)
FLASH_MODEL = "glm-5.3-flash"
random.seed(0); BOOT = 20000

rcParams.update({"font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "font.size": 11,
                 "axes.edgecolor": "#98A2AE", "axes.linewidth": 0.8, "axes.labelcolor": "#465061",
                 "xtick.color": "#465061", "ytick.color": "#465061", "text.color": "#1B2430",
                 "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150, "savefig.dpi": 300,
                 "pdf.fonttype": 42})
PRO_C, FLASH_C = "#1a6fc4", "#2fa36b"
GRID, INK, MUTED = "#D6DBE1", "#1B2430", "#465061"
AGENT = {"pi": "pi", "opencode": "OpenCode", "hermes": "Hermes"}


def load(path, model=None):
    out = defaultdict(list)
    base = None
    for r in csv.DictReader(open(path)):
        if model and r["model"] != model:
            continue
        if r["counted"] != "True" or r["status"] != "scored" or not r["holdout_auc"]:
            continue
        if r["refit_suspect"] == "True" or r["frame_stats"] == "True":
            continue
        out[r["harness"]].append(float(r["holdout_auc"]))
        base = base or (float(r["baseline_auc"]) if r.get("baseline_auc") else None)
    return out, base


A, base = load(PRO)
B, base2 = load(FLASH, FLASH_MODEL)
base = base or base2
agents = sorted(set(A) & set(B), key=lambda h: st.mean(A[h]) - st.mean(B[h]))   # smallest gain at the bottom
sd_noise = st.median([st.stdev(B[h]) for h in agents])

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.9), gridspec_kw={"width_ratios": [1.65, 1]})

# ---- left: the distributions, two rows per agent
# The gain is not annotated here on purpose: the right panel is the gain, and drawing it twice collided with itself.
for i, h in enumerate(agents):
    for j, (vals, colour, marker) in enumerate(((B[h], FLASH_C, "o"), (A[h], PRO_C, "s"))):
        y = i * 2.2 + (0.42 if j else -0.42)
        ax.plot([min(vals), max(vals)], [y, y], color=GRID, lw=4.5, solid_capstyle="round", zorder=1)
        ax.scatter(vals, [y] * len(vals), s=13, color=colour, alpha=0.55, edgecolor="none", zorder=2, marker=marker)
        ax.scatter([st.mean(vals)], [y], s=44, color=INK, edgecolor="white", linewidth=1.1, zorder=4, marker=marker)
if base:
    ax.axvline(base, color=MUTED, lw=0.9, ls=(0, (1, 2)), zorder=0)
ax.set_yticks([i * 2.2 for i in range(len(agents))])
ax.set_yticklabels([AGENT.get(h, h) for h in agents], fontsize=9.5)
ax.set_ylim(-1.15, (len(agents) - 1) * 2.2 + 1.15)
ax.set_xlabel("holdout AUC (one mark = one valid run)", fontsize=9)
ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True); ax.tick_params(labelsize=8.5)
ax.set_title("Every run of both models", fontsize=10, color=INK, loc="left", pad=8)

# ---- right: the gain against the noise
ax2.axvline(0, color=MUTED, lw=1.0)
ax2.axvspan(0, sd_noise, color=GRID, alpha=0.55, zorder=0)
for i, h in enumerate(agents):
    g = st.mean(A[h]) - st.mean(B[h])
    d = sorted(st.mean(random.choices(A[h], k=len(A[h]))) - st.mean(random.choices(B[h], k=len(B[h])))
               for _ in range(BOOT))
    lo_, hi_ = d[int(.025 * BOOT)], d[int(.975 * BOOT)]
    ax2.plot([lo_, hi_], [i, i], color=PRO_C, lw=2.0, solid_capstyle="round", zorder=2)
    ax2.scatter([g], [i], s=52, color=PRO_C, edgecolor="white", linewidth=1.1, zorder=3, marker="s")
    ax2.text(hi_ + 0.0007, i, f"+{g:.4f}", va="center", fontsize=8.5, color=INK)
ax2.set_yticks(range(len(agents))); ax2.set_yticklabels([AGENT.get(h, h) for h in agents], fontsize=9.5)
ax2.set_ylim(-0.6, len(agents) - 0.4)
ax2.set_xlabel("gain from the larger model (AUC)", fontsize=9)
ax2.text(sd_noise, len(agents) - 0.45, f"  one run-to-run SD ({sd_noise:.4f})", fontsize=8, color=MUTED, va="top")
ax2.grid(axis="x", color=GRID, lw=0.6); ax2.set_axisbelow(True); ax2.tick_params(labelsize=8.5)
ax2.set_title("The gain, with 95% interval", fontsize=10, color=INK, loc="left", pad=8)

fig.legend(handles=[Line2D([], [], marker="o", ls="", color=FLASH_C, markersize=6, label="GLM-5.3 Flash"),
                    Line2D([], [], marker="s", ls="", color=PRO_C, markersize=6, label="GLM-5.3"),
                    Line2D([], [], marker="s", ls="", color=INK, markersize=7, label="mean"),
                    Line2D([], [], color=MUTED, lw=0.9, ls=(0, (1, 2)),
                           label=f"the starting code ({base:.4f})" if base else "starting code")],
           loc="lower center", ncol=4, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.03))
fig.suptitle("A whole model tier buys about as much as running the same pairing twice",
             fontsize=11.5, color=INK, x=0.015, ha="left")
fig.tight_layout(rect=(0, 0.06, 1, 0.94))
fig.savefig(OUT / "pro_vs_flash.png", bbox_inches="tight")
fig.savefig(OUT / "pro_vs_flash.pdf", bbox_inches="tight")
print(f"wrote {OUT}/pro_vs_flash.png|pdf  (noise SD {sd_noise:.4f}; agents {', '.join(agents)})")
