"""Run one grid cell: materialize a workdir, launch the harness, enforce the wall clock, evaluate.

Two isolation modes (bench.yaml `isolation`):
  container  one throwaway Docker container per cell. It bind-mounts ONLY this cell's run directory at /cell
             (workdir, logs, harness state, prompt), runs with --cpus/--memory caps and a fresh home. It cannot
             see other cells, the data/ tree (where the private holdout lives), or the operator's dotfiles.
             The orchestrator (this process) talks to the host Docker daemon; inside docker compose that is the
             mounted /var/run/docker.sock, and BENCH_HOST_ROOT maps /app/... back to host paths for the mounts.
  process    the harness runs as a child process of `bench` on the shared filesystem. Dev/debug only.
Evaluation always happens afterwards, in the orchestrator, with pristine data and the holdout (see evaluate.py).
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from bench import ROOT
from bench.config import Config, Model
from bench.workdir import materialize

HARNESS_DIR = ROOT / "harnesses"
CELL_MOUNT = "/cell"                      # where the run dir appears inside a cell container
IMAGE_ROOT = "/app"                       # where the repo lives inside the image
IMAGE_PYTHON = f"{IMAGE_ROOT}/.venv/bin/python"


def run_dir_for(dataset: str, harness: str, model: Model, seed: int, root: Path | None = None) -> Path:
    root = (root or ROOT / "runs")
    return Path(root).resolve() / dataset / harness / model.slug / f"seed{seed}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def host_path(p: Path) -> str:
    """Translate an orchestrator path to the host path a sibling container must bind-mount."""
    host_root = os.environ.get("BENCH_HOST_ROOT")
    if host_root:
        try:
            return str(Path(host_root) / p.resolve().relative_to(ROOT.resolve()))
        except ValueError:
            pass
    return str(p.resolve())


def harness_version(harness: str) -> str | None:
    exe = shutil.which(harness)
    if not exe:
        return None
    try:
        r = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=30)
        out = (r.stdout or r.stderr).strip()
        return out.splitlines()[0][:120] if out else None
    except Exception:
        return None


def image_versions() -> dict | None:
    p = ROOT / "image_versions.json"
    try:
        return json.loads(p.read_text()) if p.exists() else None
    except Exception:
        return None


def endpoint_status(cfg: Config, model: Model) -> tuple[bool, str]:
    """Tiny chat request against the model's endpoint. (ok, detail). Used to hold the grid while a quota is exhausted."""
    import requests
    base, key = cfg.llm_endpoint(model)
    try:
        r = requests.post(f"{base}/v1/chat/completions", headers={"Authorization": f"Bearer {key}"} if key else {},
                          json={"model": model.name, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}, timeout=150)
    except requests.exceptions.ReadTimeout:
        # the endpoint accepted the request but a thinking model can take minutes to emit its first token: slow, not down
        return True, "slow (read timeout on a 1-token probe); proceeding"
    except Exception as ex:
        return False, f"unreachable: {ex.__class__.__name__}"
    if r.status_code == 200:
        return True, "ok"
    msg = ""
    try:
        msg = r.json().get("error", {}).get("message", "")
    except Exception:
        msg = r.text[:200]
    return False, f"HTTP {r.status_code}: {msg[:200]}"


def wait_for_quota(cfg: Config, model: Model, poll_seconds: int = 300, log=print) -> None:
    """Block until the endpoint accepts requests. A 429 'usage limit' from Ollama Cloud clears when the session
    window resets (or credits are added); launching cells before that only produces stalled runs."""
    ok, detail = endpoint_status(cfg, model)
    if ok and detail != "ok":
        log(f"[quota] {model.name}: {detail}")
    while not ok:
        log(f"[quota] {model.name}: {detail} -- holding the grid, re-checking every {poll_seconds // 60} min")
        time.sleep(poll_seconds)
        ok, detail = endpoint_status(cfg, model)


def docker_available() -> bool:
    if not shutil.which("docker"):
        return False
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=20).returncode == 0
    except Exception:
        return False


def cell_env(cfg: Config, *, harness: str, model: Model, dataset: str, seed: int, deadline_epoch: int,
             wall_minutes: int, max_experiments: int, paths: dict) -> dict:
    """Environment the harness script sees. `paths` are as seen from inside the cell (container or process)."""
    base_url, api_key = cfg.llm_endpoint(model)
    return {
        "PYTHONUNBUFFERED": "1",
        "BENCH_ROOT": paths["root"],
        "BENCH_HARNESS": harness,
        "BENCH_MODEL": model.name,
        "BENCH_MODEL_ENDPOINT": model.endpoint,
        "BENCH_DATASET": dataset,
        "BENCH_SEED": str(seed),
        "BENCH_WORKDIR": paths["workdir"],
        "BENCH_RUN_DIR": paths["run_dir"],
        "BENCH_LOG_DIR": paths["logs"],
        "BENCH_STATE_DIR": paths["state"],
        "BENCH_PROMPT_FILE": paths["prompt"],
        "BENCH_LLM_BASE_URL": base_url,
        "BENCH_LLM_API_KEY": api_key,
        "BENCH_WALL_SECONDS": str(wall_minutes * 60),
        "BENCH_DEADLINE_EPOCH": str(deadline_epoch),
        "BENCH_THREADS": str(cfg.threads),
        "OMP_NUM_THREADS": str(cfg.threads),
        "EXPERIMENT_TIMEOUT": str(cfg.experiment_timeout(dataset)),
        "MAX_EXPERIMENTS": str(max_experiments),
        # phase-2 compute budget: the cell-side ledger hook (bench/cell_sitecustomize.py) reads these
        "BENCH_LEDGER_DIR": paths["logs"],
        "BENCH_CPU_BUDGET_S": str(cfg.cpu_budget(dataset)),
        "BENCH_CPU_KILL_S": str(cfg.cpu_kill(dataset)),
    }


def container_cpu_seconds(name: str) -> float | None:
    """Cumulative CPU seconds of a container from the Docker Engine API over the unix socket (cgroup cpu usage).
    None if the container is gone or the socket is unavailable."""
    import http.client
    import socket

    class _Conn(http.client.HTTPConnection):
        def connect(self):
            self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.sock.settimeout(20)
            self.sock.connect(os.environ.get("DOCKER_SOCKET", "/var/run/docker.sock"))

    try:
        c = _Conn("localhost")
        c.request("GET", f"/containers/{name}/stats?stream=false&one-shot=true")
        r = c.getresponse()
        if r.status != 200:
            return None
        d = json.loads(r.read())
        return d["cpu_stats"]["cpu_usage"]["total_usage"] / 1e9
    except Exception:
        return None


def cpuset_for(slot: int, threads: int) -> str | None:
    """Pin a cell to `threads` specific cores so `nproc` inside the container reports exactly that many.
    A cgroup --cpus quota alone leaves nproc at the host count, and an agent that picks n_jobs=nproc then
    oversubscribes a throttled container (the classic 'XGBoost slower with more threads' trap). With more
    parallel slots than core groups, cells share a group round-robin: fits collide occasionally and run slower,
    but no cell ever runs more threads than it has cores."""
    total = os.cpu_count() or 0
    groups = total // threads
    if groups < 1:
        return None
    g = slot % groups                     # more slots than groups: share a group round-robin (each cell still sees `threads` CPUs)
    lo, hi = g * threads, g * threads + threads - 1
    return f"{lo}-{hi}"


def run_cell(cfg: Config, harness: str, model: Model, dataset: str, seed: int, *,
             force: bool = False, wall_minutes: int | None = None, max_experiments: int | None = None,
             research: bool | None = None, runs_root: Path | None = None, evaluate: bool = True,
             isolation: str | None = None, slot: int = 0) -> dict:
    from bench.evaluate import evaluate_run

    isolation = isolation or cfg.isolation
    if isolation not in ("container", "process"):
        raise SystemExit(f"unknown isolation mode {isolation!r} (container|process)")
    if isolation == "container" and not docker_available():
        raise SystemExit("isolation=container needs the docker CLI and a reachable daemon "
                         "(docker compose provides both). Use --isolation process only for debugging.")
    wall_minutes = wall_minutes or cfg.wall_clock(dataset)
    max_experiments = max_experiments or cfg.max_experiments
    run_dir = run_dir_for(dataset, harness, model, seed, runs_root)
    if (run_dir / "eval.json").exists() and not force:
        return {"status": "skipped", "run_dir": str(run_dir)}
    if harness != "_probe":            # the probe never calls the model; a real cell must not start into a 429
        wait_for_quota(cfg, model)
    script = HARNESS_DIR / f"{harness}.sh"
    if not script.exists():
        raise SystemExit(f"no harness runner: {script}")
    if run_dir.exists():
        shutil.rmtree(run_dir)
    (run_dir / "logs").mkdir(parents=True)
    (run_dir / "harness_state").mkdir()

    start = time.time()
    deadline = int(start) + wall_minutes * 60
    python_path = IMAGE_PYTHON if isolation == "container" else str(ROOT / ".venv" / "bin" / "python")
    cell_logs = f"{CELL_MOUNT}/logs" if isolation == "container" else str(run_dir / "logs")
    workdir = materialize(run_dir / "workdir", dataset, cfg, research=research, python_path=python_path,
                          budget_env={"MAX_EXPERIMENTS": max_experiments, "BENCH_DEADLINE_EPOCH": deadline,
                                      "EXPERIMENT_TIMEOUT": cfg.experiment_timeout(dataset), "BENCH_THREADS": cfg.threads},
                          paths_in_cell={"logs": cell_logs})
    (run_dir / "prompt.txt").write_text(cfg.prompt + "\n")

    if isolation == "container":
        paths = {"root": IMAGE_ROOT, "run_dir": CELL_MOUNT, "workdir": f"{CELL_MOUNT}/workdir",
                 "logs": f"{CELL_MOUNT}/logs", "state": f"{CELL_MOUNT}/harness_state", "prompt": f"{CELL_MOUNT}/prompt.txt"}
    else:
        paths = {"root": str(ROOT), "run_dir": str(run_dir), "workdir": str(workdir),
                 "logs": str(run_dir / "logs"), "state": str(run_dir / "harness_state"), "prompt": str(run_dir / "prompt.txt")}
    env = cell_env(cfg, harness=harness, model=model, dataset=dataset, seed=seed, deadline_epoch=deadline,
                   wall_minutes=wall_minutes, max_experiments=max_experiments, paths=paths)

    image = os.environ.get("BENCH_IMAGE", "harness-benchmark:local")
    container = f"bench-{dataset}-{harness}-{model.slug}-s{seed}-{os.getpid()}"
    meta = {
        "harness": harness, "model": model.name, "model_endpoint": model.endpoint, "dataset": dataset, "seed": seed,
        "wall_clock_minutes": wall_minutes, "max_experiments": max_experiments,
        "experiment_timeout_seconds": cfg.experiment_timeout(dataset), "threads": cfg.threads,
        "cpu_budget_seconds": cfg.cpu_budget(dataset), "cpu_kill_seconds": cfg.cpu_kill(dataset),
        "research_allowed": cfg.research_allowed if research is None else research,
        "llm_base_url": env["BENCH_LLM_BASE_URL"], "started": _now(),
        "isolation": isolation,
        "container": {"name": container, "image": image, "cpus": cfg.cell_cpus, "memory": cfg.cell_memory,
                      "cpuset": cpuset_for(slot, cfg.threads)} if isolation == "container" else None,
        "harness_version": harness_version(harness) if isolation == "process" else None,
        "in_docker": bool(os.environ.get("BENCH_IN_DOCKER")),
        "image_versions": image_versions(),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    if isolation == "container":
        cmd = ["docker", "run", "--rm", "--init", "--entrypoint", "bash", "--name", container,
               "--cpus", str(cfg.cell_cpus), "--memory", str(cfg.cell_memory)]
        cpuset = cpuset_for(slot, cfg.threads)
        if cpuset:
            cmd += ["--cpuset-cpus", cpuset]
        cmd += [
               "--security-opt", "no-new-privileges",
               "-v", f"{host_path(run_dir)}:{CELL_MOUNT}",
               "-w", paths["workdir"], "-u", "1000:1000"]
        for k, v in env.items():
            cmd += ["-e", f"{k}={v}"]
        cmd += [image, f"{IMAGE_ROOT}/harnesses/{harness}.sh"]
        popen_env = dict(os.environ)
    else:
        cmd = ["bash", str(script)]
        popen_env = {**os.environ, **env, "PATH": f"{ROOT / '.venv' / 'bin'}:{os.environ.get('PATH', '')}"}

    with open(run_dir / "logs" / "harness.stdout", "wb") as out, open(run_dir / "logs" / "harness.stderr", "wb") as err:
        proc = subprocess.Popen(cmd, cwd=workdir if isolation == "process" else None, env=popen_env,
                                stdout=out, stderr=err, stdin=subprocess.DEVNULL, start_new_session=True)
        timed_out = stalled = cpu_killed = blowup = False
        max_bytes = cfg.cell_max_gb * 1e9
        last_du = 0.0
        cpu_container = None
        hard_deadline = time.time() + wall_minutes * 60 + 90   # 90s grace so a harness that watches the deadline can wrap up
        stall_s = cfg.stall_minutes * 60
        cpu_kill = cfg.cpu_kill(dataset) if isolation == "container" else 0
        while True:
            try:
                proc.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                pass
            if isolation == "container":
                c = container_cpu_seconds(container)
                if c is not None:
                    cpu_container = c
            if time.time() > hard_deadline:
                timed_out = True
            elif stall_s and time.time() - _last_activity(run_dir) > stall_s:
                stalled = True
            elif cpu_kill and cpu_container is not None and cpu_container > cpu_kill:
                cpu_killed = True                                   # hard backstop: the ledger hook was bypassed or overrun
                (run_dir / "logs" / "cpu_kill.txt").write_text(f"container CPU {cpu_container:.0f}s > cap {cpu_kill}s at {_now()}\n")
            elif max_bytes and time.time() - last_du > 120 and _dir_bytes(run_dir) > max_bytes:
                blowup = True                                       # runaway writes (a 135 GB tool log filled a disk once)
                (run_dir / "logs" / "disk_kill.txt").write_text(f"run dir > {cfg.cell_max_gb} GB at {_now()}\n")
            else:
                if time.time() - last_du > 120:
                    last_du = time.time()
                continue
            if isolation == "container":
                subprocess.run(["docker", "kill", container], capture_output=True, timeout=60)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    _kill_group(proc)
            else:
                _kill_group(proc)
            break
    if isolation == "container":
        # the harness log line with `<tool> --version` is the only version record we get from inside the cell
        subprocess.run(["docker", "rm", "-f", container], capture_output=True, timeout=60)
    # If the endpoint is refusing requests when the cell ends, the cell almost certainly stalled on quota/outage
    # (harness retry loops are silent in some CLIs). Recorded here so evaluate can flag it regardless of log format.
    ok, detail = endpoint_status(cfg, model)
    meta.update({
        "ended": _now(), "wall_seconds": round(time.time() - start, 1),
        "exit_code": proc.returncode, "timed_out": timed_out, "stalled": stalled, "disk_blowup": blowup,
        "cpu_killed": cpu_killed, "cpu_seconds_container": round(cpu_container, 1) if cpu_container is not None else None,
        "endpoint_at_end": detail if not ok else "ok",
    })
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    result = {"status": "ran", "run_dir": str(run_dir), **meta}
    if evaluate:
        result["eval"] = evaluate_run(run_dir, cfg)
    return result


def _dir_bytes(run_dir: Path) -> int:
    """Total size of the run dir (fast: du -s). Used to catch runaway writes such as a tool-output log that never stops."""
    try:
        out = subprocess.run(["du", "-sk", str(run_dir)], capture_output=True, text=True, timeout=60).stdout.split()[0]
        return int(out) * 1024
    except Exception:
        return 0


def _last_activity(run_dir: Path) -> float:
    """Most recent write in the cell: the mirrored experiments.tsv, anything in the workdir, or the harness logs."""
    latest = (run_dir / "meta.json").stat().st_mtime
    for p in [run_dir / "logs" / "experiments.tsv", run_dir / "logs" / "harness.stdout", run_dir / "logs" / "harness.stderr"]:
        if p.exists():
            latest = max(latest, p.stat().st_mtime)
    wd = run_dir / "workdir"
    if wd.exists():
        for p in wd.iterdir():
            if p.name != ".git":
                try:
                    latest = max(latest, p.stat().st_mtime)
                except OSError:
                    pass
    return latest


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=10)
