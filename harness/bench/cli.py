"""bench: command-line entry point.

  bench doctor                         check binaries, keys, venv, datasets
  bench models [--check]               list models on the configured Ollama endpoint
  bench prepare <dataset|all>          build data/prepared/<dataset>
  bench baseline <dataset|all>         holdout score of the untouched template train.py
  bench run -H claude -M kimi-k3 -D airline [-s 1] [--minutes 10 --experiments 5]
  bench grid [--dry-run] [--parallel N] [-H ...] [-M ...] [-D ...] [--seeds 1 2]
  bench eval <run_dir|all> [--all-commits]
  bench report
  bench probe [-D airline]               launch a cell exactly like a real one and print what it can see
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from pathlib import Path

from bench import ROOT
from bench.config import Config, load_config


def cmd_doctor(cfg: Config, a) -> int:
    ok = True
    print("harness binaries:")
    for h in cfg.harnesses:
        p = shutil.which(h)
        print(f"  {h:10s} {'OK  ' + p if p else 'MISSING'}")
        ok &= bool(p) or h == "openclaw"
    print("python env:")
    py = ROOT / ".venv" / "bin" / "python"
    print(f"  .venv      {'OK' if py.exists() else 'MISSING (run: uv sync)'}")
    print("credentials / endpoints:")
    o = cfg.raw.get("ollama", {})
    key = os.environ.get(o.get("api_key_env", "OLLAMA_API_KEY"))
    print(f"  {o.get('api_key_env', 'OLLAMA_API_KEY'):16s} {'set' if key else 'NOT SET (export it; needed for endpoint=cloud models)'}")
    needs_local = any(m.endpoint == "local" for m in cfg.models)
    if needs_local:
        try:
            import requests
            r = requests.get(os.environ.get("OLLAMA_LOCAL_URL", "http://localhost:11434") + "/api/tags", timeout=3)
            names = {m["name"] for m in r.json().get("models", [])}
            print(f"  local ollama     running, {len(names)} models")
            for m in cfg.models:
                if m.endpoint == "local":
                    print(f"    {m.name:28s} {'pulled' if m.name in names or m.name + ':latest' in names else 'NOT PULLED (ollama pull ' + m.name + ')'}")
        except Exception as e:
            print(f"  local ollama     NOT RUNNING ({e.__class__.__name__}); needed for endpoint=local models")
    print("datasets:")
    for d in cfg.datasets:
        p = ROOT / "data" / "prepared" / d / "meta.json"
        print(f"  {d:10s} {'prepared' if p.exists() else 'NOT PREPARED (bench prepare ' + d + ')'}")
    return 0 if ok else 1


def cmd_models(cfg: Config, a) -> int:
    import requests
    base, key = cfg.llm_endpoint(cfg.model("__cloud__"))
    r = requests.get(f"{base}/api/tags", headers={"Authorization": f"Bearer {key}"} if key else {}, timeout=20)
    r.raise_for_status()
    names = sorted(m["name"] for m in r.json().get("models", []))
    for n in names:
        print(n)
    if a.check:
        print("\nconfigured cloud models:")
        for m in cfg.models:
            if m.endpoint == "cloud":
                hit = m.name in names or f"{m.name}:latest" in names
                print(f"  {m.name:28s} {'OK' if hit else 'NOT FOUND on endpoint'}")
    return 0


def cmd_prepare(cfg: Config, a) -> int:
    targets = cfg.datasets if a.dataset == "all" else [a.dataset]
    for d in targets:
        script = ROOT / "datasets" / f"prepare_{d}.py"
        if not script.exists():
            print(f"no preparer for {d}: {script}")
            return 1
        print(f"== preparing {d} ==")
        r = subprocess.run([str(ROOT / ".venv" / "bin" / "python"), str(script)], cwd=ROOT / "datasets")
        if r.returncode:
            return r.returncode
    return 0


def cmd_baseline(cfg: Config, a) -> int:
    from bench.evaluate import baseline_score
    for d in (cfg.datasets if a.dataset == "all" else [a.dataset]):
        res = baseline_score(d, cfg, force=a.force)
        print(f"{d:10s} baseline holdout AUC={res.get('auc')}  AP={res.get('ap')}  ({res.get('seconds')}s)  {res.get('error', '')}")
    return 0


def cmd_run(cfg: Config, a) -> int:
    from bench.runner import run_cell
    res = run_cell(cfg, a.harness, cfg.model(a.model), a.dataset, a.seed, force=a.force,
                   wall_minutes=a.minutes, max_experiments=a.experiments, research=a.research,
                   runs_root=Path(a.runs_root) if a.runs_root else None, isolation=a.isolation)
    _print_result(res)
    return 0


def _print_result(res: dict) -> None:
    e = res.get("eval") or {}
    print(json.dumps({k: res.get(k) for k in ("status", "run_dir", "exit_code", "timed_out", "wall_seconds", "error", "cell", "cpu_killed", "cpu_seconds_container")}
                     | {k: e.get(k) for k in ("holdout_auc", "baseline_holdout_auc", "best_eval_auc", "n_experiments", "n_commits", "violations", "holdout_error", "cpu_seconds_python", "cpu_seconds_uncounted", "cpu_refusals", "cpu_over_budget_events", "fits_total")}, indent=None))


def cmd_grid(cfg: Config, a) -> int:
    from bench.runner import run_cell, run_dir_for
    harnesses = a.harness or cfg.harnesses
    models = [cfg.model(m) for m in (a.model or [m.name for m in cfg.models])]
    datasets = a.dataset or cfg.datasets
    seeds = a.seeds or cfg.seeds
    runs_root = Path(a.runs_root) if getattr(a, "runs_root", None) else None   # e.g. /app/runs_smoke/phase2 for a smoke grid
    # model-major, cheapest model first: the cheap rows of the grid complete on a fraction of the provider quota,
    # and the expensive rows (kimi-k3, glm-5.3) come last where the quota hold paces them. Unknown prices sort last.
    def est_cost(m):
        p = cfg.price(m.name)
        return (p["input"] * 0.4 + p["cached"] * 0.6) * 8 + p["output"] * 0.06 if p else float("inf")
    models = sorted(models, key=est_cost)
    cells = [(s, d, h, m) for s, d, m, h in product(seeds, datasets, models, harnesses)]
    def done(c):
        p = run_dir_for(c[1], c[2], c[3], c[0], runs_root) / "eval.json"
        if not p.exists():
            return False
        try:
            return not json.loads(p.read_text()).get("provider_error")   # quota/outage cells are not done
        except Exception:
            return False
    todo = [c for c in cells if a.force or not done(c)]
    print(f"{len(cells)} cells, {len(todo)} to run, parallel={a.parallel or cfg.parallel}, isolation={a.isolation or cfg.isolation}, "
          f"pace={cfg.max_cells_per_hour or 'unlimited'} cells/h, budget={cfg.max_experiments} experiments, ceiling " + "/".join(f"{d}={cfg.wall_clock(d)}min" for d in datasets) + f", stall={cfg.stall_minutes}min")
    if a.dry_run:
        for s, d, h, m in todo:
            print(f"  seed{s} {d:8s} {h:9s} {m.name}")
        return 0
    import queue, threading, time as _time
    n_par = a.parallel or cfg.parallel
    slots = queue.Queue()
    for i in range(n_par):
        slots.put(i)
    pace_lock, last_start = threading.Lock(), [0.0]
    min_gap = 3600.0 / cfg.max_cells_per_hour if cfg.max_cells_per_hour else 0.0

    def one(c):
        s, d, h, m = c
        slot = slots.get()
        try:
            if min_gap:
                with pace_lock:                       # never start cells faster than max_cells_per_hour
                    wait = last_start[0] + min_gap - _time.time()
                    if wait > 0:
                        print(f"[pace] holding {wait/60:.1f} min before {d}/{h}/{m.name}/seed{s}")
                        _time.sleep(wait)
                    last_start[0] = _time.time()
            return run_cell(cfg, h, m, d, s, force=True, isolation=a.isolation, slot=slot, runs_root=runs_root)
        except Exception as ex:  # never let one cell kill the grid
            return {"status": "error", "error": repr(ex), "cell": f"{d}/{h}/{m.name}/seed{s}"}
        finally:
            slots.put(slot)
    with ThreadPoolExecutor(max_workers=n_par) as ex:
        futs = {ex.submit(one, c): c for c in todo}
        for f in as_completed(futs):
            _print_result(f.result())
    from bench.report import build_report
    print(f"report: {build_report()}")
    return 0


def cmd_eval(cfg: Config, a) -> int:
    from bench.evaluate import evaluate_run
    if a.run_dir == "all":
        dirs = sorted(p.parent for p in (ROOT / "runs").glob("*/*/*/seed*/meta.json"))
    else:
        dirs = [Path(a.run_dir)]
    for d in dirs:
        r = evaluate_run(d, cfg, all_commits=a.all_commits, keep_scores=a.keep_scores, best_commit=a.best_commit)
        print(f"{d}: holdout_auc={r.get('holdout_auc')} best_eval={r.get('best_eval_auc')} n_exp={r.get('n_experiments')} viol={r.get('violations')} err={r.get('holdout_error')}")
    return 0


def cmd_probe(cfg: Config, a) -> int:
    from bench.runner import run_cell
    res = run_cell(cfg, "_probe", cfg.model(a.model), a.dataset, 0, force=True, wall_minutes=3, max_experiments=1,
                   runs_root=ROOT / "runs_smoke" / "probe", evaluate=False, isolation=a.isolation)
    rd = Path(res["run_dir"])
    print(f"isolation={res.get('isolation')}  exit={res.get('exit_code')}\n")
    print((rd / "logs" / "harness.stdout").read_text())
    err = (rd / "logs" / "harness.stderr").read_text().strip()
    if err:
        print("[stderr]", err[-800:])
    return 0


def cmd_report(cfg: Config, a) -> int:
    from bench.report import build_report
    p = build_report(Path(a.runs_root) if a.runs_root else None)
    print(p.read_text())
    return 0


def load_dotenv(path: Path) -> None:
    """Minimal .env loader (KEY=VALUE, # comments); never overrides variables already set."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.split(" #", 1)[0].strip().strip('"').strip("'")
        if v:
            os.environ.setdefault(k.strip(), v)


def main(argv=None) -> int:
    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser(prog="bench", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="path to bench.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor")
    p = sub.add_parser("models"); p.add_argument("--check", action="store_true")
    p = sub.add_parser("prepare"); p.add_argument("dataset")
    p = sub.add_parser("baseline"); p.add_argument("dataset"); p.add_argument("--force", action="store_true")
    p = sub.add_parser("run")
    p.add_argument("-H", "--harness", required=True); p.add_argument("-M", "--model", required=True)
    p.add_argument("-D", "--dataset", required=True); p.add_argument("-s", "--seed", type=int, default=1)
    p.add_argument("--minutes", type=int); p.add_argument("--experiments", type=int)
    p.add_argument("--research", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--force", action="store_true"); p.add_argument("--runs-root")
    p.add_argument("--isolation", choices=["container", "process"])
    p = sub.add_parser("grid")
    p.add_argument("-H", "--harness", nargs="*"); p.add_argument("-M", "--model", nargs="*"); p.add_argument("-D", "--dataset", nargs="*")
    p.add_argument("--seeds", nargs="*", type=int); p.add_argument("--parallel", type=int)
    p.add_argument("--dry-run", action="store_true"); p.add_argument("--force", action="store_true")
    p.add_argument("--isolation", choices=["container", "process"]); p.add_argument("--runs-root")
    p = sub.add_parser("probe"); p.add_argument("-D", "--dataset", default="airline"); p.add_argument("-M", "--model", default="gpt-oss:120b")
    p.add_argument("--isolation", choices=["container", "process"])
    p = sub.add_parser("eval"); p.add_argument("run_dir"); p.add_argument("--all-commits", action="store_true")
    p.add_argument("--best-commit", action="store_true", help="also score the best-eval committed experiment (what the agent found, vs HEAD = what it delivered)")
    p.add_argument("--keep-scores", action="store_true", help="refresh usage/violations but keep holdout scores from the existing eval.json")
    p = sub.add_parser("report"); p.add_argument("--runs-root")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    return {"doctor": cmd_doctor, "models": cmd_models, "prepare": cmd_prepare, "baseline": cmd_baseline,
            "run": cmd_run, "grid": cmd_grid, "eval": cmd_eval, "report": cmd_report, "probe": cmd_probe}[a.cmd](cfg, a)


if __name__ == "__main__":
    sys.exit(main())
