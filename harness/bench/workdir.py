"""Materialize an agent workdir from task_template + a prepared dataset."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from string import Template

from bench import ROOT
from bench.config import Config

TEMPLATE = ROOT / "task_template"
PREPARED = ROOT / "data" / "prepared"
PROTECTED = ["run_experiment.sh", "validate.sh", "validate.py", "task.json", "data/train.csv", "data/eval.csv"]  # all tracked
GIT_ENV = {
    "GIT_AUTHOR_NAME": "bench", "GIT_AUTHOR_EMAIL": "bench@localhost",
    "GIT_COMMITTER_NAME": "bench", "GIT_COMMITTER_EMAIL": "bench@localhost",
    "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1",
}


def dataset_meta(dataset: str) -> dict:
    p = PREPARED / dataset / "meta.json"
    if not p.exists():
        raise SystemExit(f"dataset '{dataset}' not prepared ({p} missing). Run: bench prepare {dataset}")
    return json.loads(p.read_text())


def render_program(cfg: Config, meta: dict, research: bool, dataset: str) -> str:
    section = (TEMPLATE / ("research_section.md" if research else "no_research_section.md")).read_text()
    t = Template((TEMPLATE / "program.md").read_text())
    return t.safe_substitute(
        DATASET_NAME=meta["name"],
        DATASET_DESCRIPTION=meta["description"],
        TARGET=meta["target"],
        POSITIVE_LABEL=meta["positive_label"],
        ID_COLUMNS=", ".join(meta.get("id_columns") or []) or "(none)",
        SPLIT_DESCRIPTION=meta["split"],
        MAX_EXPERIMENTS=cfg.max_experiments,
        WALL_MINUTES=cfg.wall_clock(dataset),
        EXPERIMENT_TIMEOUT=cfg.experiment_timeout(dataset),
        CELL_MEMORY=str(cfg.raw.get("cell_memory", "8g")).upper().replace("G", " GB"),   # disclosed to the agent since 2026-09-12
        THREADS=cfg.threads,
        CPU_BUDGET_LINE=cpu_budget_line(cfg, dataset),
        RESEARCH_SECTION=section,
    )


def cpu_budget_line(cfg: Config, dataset: str) -> str:
    """The phase-2 compute rule as the agent reads it; empty when the budget is off (phase-1 behaviour)."""
    b, k = cfg.cpu_budget(dataset), cfg.cpu_kill(dataset)
    if not b:
        return ""
    return (f"- **CPU: {b:,} CPU-seconds of Python compute** for the whole cell, measured by the interpreter across every\n"
            f"  Python process you start (`./run_experiment.sh` runs, your own scripts, one-liners). `./run_experiment.sh`\n"
            f"  prints how much is used. When it is spent, Python refuses to start, so finalize before that. Above\n"
            f"  {k:,} CPU-seconds the container is stopped outright.\n")


def materialize(workdir: Path, dataset: str, cfg: Config, research: bool | None = None,
                budget_env: dict | None = None, python_path: str | None = None,
                paths_in_cell: dict | None = None) -> Path:
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    meta = dataset_meta(dataset)
    research = cfg.research_allowed if research is None else research

    for f in TEMPLATE.iterdir():
        if f.name in ("program.md", "research_section.md", "no_research_section.md", "data"):
            continue
        if f.is_file():
            shutil.copy2(f, workdir / f.name)
    (workdir / "program.md").write_text(render_program(cfg, meta, research, dataset))
    task = {k: meta[k] for k in ("name", "description", "target", "positive_label", "id_columns", "split", "columns", "rows")}
    (workdir / "task.json").write_text(json.dumps(task, indent=2))
    (workdir / "data").mkdir()
    for name in ("train.csv", "eval.csv"):
        shutil.copy2(PREPARED / dataset / "public" / name, workdir / "data" / name)
    # data/ is TRACKED in the cell's git repo: `git clean -fdx` cannot remove it and `git reset --hard` restores it
    # (a pilot agent wiped its own dataset with `git clean -fdx`). The evaluator re-hashes it against the pristine copy.
    # Budget + interpreter live in <run dir>/bench.env, OUTSIDE the repo, readable by run_experiment.sh even when the
    # harness runs commands in a clean environment; every experiments.tsv row is mirrored to <run dir>/logs/ as well.
    run_dir = workdir.parent
    (run_dir / "logs").mkdir(exist_ok=True)
    mirror = (paths_in_cell or {}).get("logs", str(run_dir / "logs")) + "/experiments.tsv"
    env = {"BENCH_PYTHON": python_path or str(ROOT / ".venv" / "bin" / "python"), "EXPERIMENT_TIMEOUT": cfg.experiment_timeout(dataset),
           "MAX_EXPERIMENTS": cfg.max_experiments, "BENCH_THREADS": cfg.threads, "BENCH_DEADLINE_EPOCH": 0,
           "BENCH_EXPERIMENTS_MIRROR": mirror,
           "BENCH_LEDGER_DIR": (paths_in_cell or {}).get("logs", str(run_dir / "logs")),   # cpu_ledger.tsv / fits.tsv live here
           "BENCH_CPU_BUDGET_S": cfg.cpu_budget(dataset), "BENCH_CPU_KILL_S": cfg.cpu_kill(dataset),
           **(budget_env or {})}
    (run_dir / "bench.env").write_text("".join(f"{k}={v}\n" for k, v in env.items()))

    def git(*args):
        subprocess.run(["git", *args], cwd=workdir, check=True, capture_output=True, env={**GIT_ENV, "PATH": "/usr/bin:/bin:/opt/homebrew/bin:/usr/local/bin"})

    git("init", "-q", "-b", "experiment")
    git("add", "-A")
    git("commit", "-q", "-m", "baseline")
    return workdir
