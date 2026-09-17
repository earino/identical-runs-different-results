"""Post-run evaluation of a cell: holdout AUC of the final train.py, trajectory stats, protocol checks, usage."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

from bench import ROOT
from bench.config import Config
from bench.usage import extract_usage
from bench.workdir import GIT_ENV, PREPARED, PROTECTED, TEMPLATE

BASELINES = ROOT / "results" / "baselines"


def _git(workdir: Path, *args: str) -> str:
    r = subprocess.run(["git", *args], cwd=workdir, capture_output=True, text=True, env={**GIT_ENV, "PATH": os.environ["PATH"]})
    return r.stdout.strip()


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def protocol_violations(workdir: Path, dataset: str) -> list[str]:
    bad = []
    root_commit = _git(workdir, "rev-list", "--max-parents=0", "HEAD")
    # protocol files: the working copy must equal the copy committed in this cell's own baseline commit (the template
    # evolves between runs, so the live template is not the reference)
    for rel in ("run_experiment.sh", "validate.sh", "validate.py", "task.json"):
        p = workdir / rel
        ref = _git(workdir, "show", f"{root_commit}:{rel}") if root_commit else ""
        if not p.exists():
            if ref:
                bad.append(f"missing:{rel}")
        elif ref and p.read_text(errors="replace").rstrip("\n") != ref.rstrip("\n"):   # _git strips trailing newlines
            bad.append(f"modified:{rel}")
    # data: must equal the prepared public split (tracked in the cell repo since the hardening; hash either way)
    for rel, src in (("data/train.csv", PREPARED / dataset / "public" / "train.csv"),
                     ("data/eval.csv", PREPARED / dataset / "public" / "eval.csv")):
        p = workdir / rel
        if not p.exists():
            bad.append(f"missing:{rel}")
        elif _sha(p) != _sha(src):
            bad.append(f"modified:{rel}")
    # any tracked file other than train.py / FINAL.md changed since baseline?
    if root_commit:
        changed = _git(workdir, "diff", "--name-only", root_commit, "HEAD").splitlines()
        for f in changed:
            if f not in ("train.py", "FINAL.md") and f"modified:{f}" not in bad:
                bad.append(f"modified-tracked:{f}")
    # the final train.py is executed by the scorer next to the holdout, so scan the lines the AGENT ADDED
    # (diff baseline..HEAD, code lines only) for filesystem/network escape idioms
    if root_commit:
        diff = _git(workdir, "diff", f"{root_commit}", "HEAD", "--", "train.py")
        added = [l[1:] for l in diff.splitlines() if l.startswith("+") and not l.startswith("+++")]
        code = "\n".join(l for l in added if not l.strip().startswith("#"))
        for token in ("holdout", "private/", "/cell/", "/app/", "os.walk", "glob(", "subprocess", "urllib", "requests.", "socket", "../"):
            if token in code:
                bad.append(f"suspicious-train-py:{token}")
    log_dir = workdir.parent / "logs"
    for name in ("harness.stdout", "harness.stderr"):
        p = log_dir / name
        if p.exists():
            t = p.read_text(errors="replace")
            if "private/holdout" in t or "/private/" in t and "holdout.csv" in t:
                bad.append(f"holdout-path-mentioned:{name}")
            if "pip install" in t or "uv add" in t or "uv pip install" in t or "conda install" in t:
                bad.append(f"package-install-attempt:{name}")
    return bad


# substrings that only occur in provider-side failure messages (bare status numbers like "429" would also match
# token counts and hashes in event logs, so they are anchored to the words that accompany them)
# Provider-side failure signatures. Bare "quota" and "billing" were removed on 2026-09-15: agents write those words in
# their own reasoning ("whether the process counts against some other quota"), which flagged two clean phase-2 cells.
PROVIDER_ERROR_PATTERNS = ("usage limit", "rate limit", "rate_limit", "ratelimit", "too many requests", "http 429", "status 429",
                           "code 429", "error 429", "status_code=429", "insufficient credits", "usage credits",
                           "quota exceeded", "insufficient_quota", "exceeded your current quota", "out of quota",
                           "payment required", "http 402", "billing hard limit", "overloaded_error", "service unavailable",
                           "bad gateway", "http 503", "http 502")


def provider_error(log_dir: Path) -> str | None:
    """First provider-side failure signature (quota, rate limit, outage) found in the harness logs, else None.
    Such a cell did not get a fair run and `bench grid` re-runs it instead of counting it."""
    for name in ("harness.stdout", "harness.stderr"):
        p = log_dir / name
        if not p.exists():
            continue
        text = p.read_text(errors="replace").lower()
        for pat in PROVIDER_ERROR_PATTERNS:
            i = text.find(pat)
            if i >= 0:
                return text[max(0, i - 60):i + 100].replace("\n", " ")
        m = QUOTED_429.search(text)
        if m:
            return text[max(0, m.start() - 60):m.start() + 100].replace("\n", " ")
    return None


# 429 only as a JSON status/code field or as an HTTP reason line. A bare quoted `"429` matched text that merely starts with
# 429 in event logs: commit hashes in tool output ("text":"429bb52 baseline") and streamed model tokens ("delta":"429),
# slower", from an AUC like 0.7429), flagging two clean run-variance cells on 2026-09-15.
QUOTED_429 = re.compile(r'"(?:status|code|status_code|statuscode|error_code|http_status)"\s*:\s*"?429(?![0-9a-z])'
                        r'|"429 (?:too many|rate|client error|resource.exhausted)')


def read_experiments(workdir: Path) -> list[dict]:
    """The mirror in <run dir>/logs/ is authoritative (git cannot touch it); fall back to the workdir copy."""
    p = workdir.parent / "logs" / "experiments.tsv"
    if not p.exists():
        p = workdir / "experiments.tsv"
    if not p.exists():
        return []
    with open(p, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    for r in rows:
        try:
            r["eval_auc"] = float(r.get("eval_auc") or 0)
        except ValueError:
            r["eval_auc"] = 0.0
    return rows


def score(workdir: Path, dataset: str, cfg: Config, commit: str = "HEAD") -> dict:
    """Score train.py at <commit> on the private holdout, in a throwaway copy with pristine data/."""
    meta = json.loads((PREPARED / dataset / "meta.json").read_text())
    holdout = PREPARED / dataset / "private" / "holdout.csv"
    with tempfile.TemporaryDirectory(prefix="bench-eval-") as tmp:
        tmpd = Path(tmp)
        archive = tmpd / "src.tar"
        with open(archive, "wb") as f:
            subprocess.run(["git", "archive", "--format=tar", commit], cwd=workdir, stdout=f, check=True,
                           env={**GIT_ENV, "PATH": os.environ["PATH"]})
        evald = tmpd / "w"
        evald.mkdir()
        with tarfile.open(archive) as t:
            t.extractall(evald, filter="data")
        (evald / "data").mkdir(exist_ok=True)
        for name in ("train.csv", "eval.csv"):
            shutil.copy2(PREPARED / dataset / "public" / name, evald / "data" / name)
        env = {**os.environ, "PATH": f"{ROOT / '.venv' / 'bin'}:{os.environ['PATH']}",
               "BENCH_THREADS": str(cfg.threads), "OMP_NUM_THREADS": str(cfg.threads), "PYTHONPATH": str(ROOT)}
        # the final model may sit at the per-experiment timeout and the holdout is up to 10x the eval set:
        # allow training + a large predict (a 2000-tree airline model needed > 480 s once)
        timeout = 10 * cfg.experiment_timeout(dataset) + 600
        try:
            r = subprocess.run(
                [str(ROOT / ".venv" / "bin" / "python"), "-m", "bench.score_holdout", str(holdout), meta["target"], json.dumps(meta["positive_label"])],
                cwd=evald, env=env, capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return {"error": f"holdout scoring timed out after {timeout}s"}
        last = (r.stdout.strip().splitlines() or [""])[-1]
        try:
            return json.loads(last)
        except Exception:
            return {"error": f"scorer failed (exit {r.returncode}): {r.stderr.strip()[-2000:]}"}


def baseline_score(dataset: str, cfg: Config, force: bool = False) -> dict:
    """Holdout score of the untouched template train.py for a dataset (cached under results/baselines/)."""
    BASELINES.mkdir(parents=True, exist_ok=True)
    p = BASELINES / f"{dataset}.json"
    if p.exists() and not force:
        return json.loads(p.read_text())
    from bench.workdir import materialize
    with tempfile.TemporaryDirectory(prefix="bench-baseline-") as tmp:
        wd = materialize(Path(tmp) / "workdir", dataset, cfg)
        res = score(wd, dataset, cfg)
    res["dataset"] = dataset
    p.write_text(json.dumps(res, indent=2))
    return res


def cost_usd(usage: dict, price: dict | None) -> float | None:
    """Dollar cost of a cell from its split token counts and the model's per-million prices."""
    if not price or usage.get("tokens_in") is None:
        return None
    unc = usage.get("tokens_in_uncached") or 0
    cached = usage.get("tokens_in_cached") or 0
    out = usage.get("tokens_out") or 0
    return round((unc * price["input"] + cached * price["cached"] + out * price["output"]) / 1e6, 4)


def cpu_summary(run_dir: Path) -> dict:
    """Phase-2 compute accounting from the cell-side ledger (logs/cpu_ledger.tsv) and fit log (logs/fits.tsv).
    kind: counted = launched by run_experiment.sh, uncounted = the agent's own Python, exempt = validate.sh."""
    out = {"cpu_seconds_python": None, "cpu_seconds_counted": None, "cpu_seconds_uncounted": None,
           "cpu_seconds_exempt": None, "python_processes": None, "cpu_refusals": None, "cpu_over_budget_events": None,
           "fits_total": None, "fits_counted": None, "fits_uncounted": None}
    led = run_dir / "logs" / "cpu_ledger.tsv"
    if led.exists():
        by = {"counted": 0.0, "uncounted": 0.0, "exempt": 0.0}
        n = refused = over = 0
        for line in led.read_text(errors="replace").splitlines()[1:]:
            parts = line.split("\t")
            if len(parts) < 5:
                continue
            try:
                cpu = float(parts[2])
            except ValueError:
                continue
            kind, event = parts[3], parts[4]
            by[kind] = by.get(kind, 0.0) + cpu
            n += 1
            refused += event == "refused"
            over += event == "over-budget"
        out.update(cpu_seconds_python=round(by["counted"] + by["uncounted"] + by["exempt"], 1),
                   cpu_seconds_counted=round(by["counted"], 1), cpu_seconds_uncounted=round(by["uncounted"], 1),
                   cpu_seconds_exempt=round(by["exempt"], 1), python_processes=n, cpu_refusals=refused,
                   cpu_over_budget_events=over)
    fits = run_dir / "logs" / "fits.tsv"
    if fits.exists():
        rows = [l.split("\t") for l in fits.read_text(errors="replace").splitlines()[1:]]
        rows = [r for r in rows if len(r) >= 3]
        out.update(fits_total=len(rows), fits_counted=sum(r[2] == "counted" for r in rows),
                   fits_uncounted=sum(r[2] == "uncounted" for r in rows))
    return out


def evaluate_run(run_dir: Path, cfg: Config, all_commits: bool = False, keep_scores: bool = False,
                 best_commit: bool = False) -> dict:
    """keep_scores: reuse holdout_auc/ap from the existing eval.json instead of re-running train.py. Scores must come
    from one platform (the Linux image on the benchmark box: XGBoost results differ slightly across CPUs/arches);
    use this when refreshing usage/violations elsewhere."""
    meta = json.loads((run_dir / "meta.json").read_text())
    prev = json.loads((run_dir / "eval.json").read_text()) if keep_scores and (run_dir / "eval.json").exists() else None
    workdir = run_dir / "workdir"
    dataset = meta["dataset"]
    head = _git(workdir, "rev-parse", "--short", "HEAD")
    log = _git(workdir, "log", "--format=%h%x09%s", "--reverse").splitlines()
    commits = [{"commit": l.split("\t")[0], "message": l.split("\t", 1)[1] if "\t" in l else ""} for l in log if l]
    exps = read_experiments(workdir)
    ok = [e for e in exps if e.get("status") == "ok"]
    best_eval = max((e["eval_auc"] for e in ok), default=None)
    # eval AUC of the final model: the newest commit (walking back from HEAD, which is usually the "final" FINAL.md
    # commit with no experiment row) that has an ok experiment row
    final_eval = None
    for c in reversed(commits):
        rows_c = [e for e in ok if e.get("commit", "").startswith(c["commit"])]
        if rows_c:
            final_eval = rows_c[-1]["eval_auc"]
            break
    if prev is not None and prev.get("holdout_auc") is not None:
        holdout = {"auc": prev["holdout_auc"], "ap": prev.get("holdout_ap")}
    elif prev is not None and prev.get("holdout_error"):
        holdout = {"error": prev["holdout_error"]}
    else:
        holdout = score(workdir, dataset, cfg) if commits else {"error": "no commits"}
    base = baseline_score(dataset, cfg)
    usage = extract_usage(meta["harness"], run_dir / "logs")
    usage["cost_usd"] = cost_usd(usage, cfg.price(meta["model"]))
    usage["cached_share"] = (round(usage["tokens_in_cached"] / usage["tokens_in"], 3)
                             if usage.get("tokens_in") else None)
    result = {
        **meta,
        "head": head,
        "n_commits": max(len(commits) - 1, 0),          # kept experiments (excluding the baseline commit)
        "n_experiments": len(exps),
        "n_ok": len(ok),
        "n_crash": sum(e.get("status") == "crash" for e in exps),
        "n_timeout": sum(e.get("status") == "timeout" for e in exps),
        "best_eval_auc": best_eval,
        "final_eval_auc": final_eval,
        "holdout_auc": holdout.get("auc"),
        "holdout_ap": holdout.get("ap"),
        "holdout_error": holdout.get("error"),
        "baseline_holdout_auc": base.get("auc"),
        "delta_vs_baseline": (holdout["auc"] - base["auc"]) if holdout.get("auc") is not None and base.get("auc") is not None else None,
        "generalization_gap": (final_eval - holdout["auc"]) if final_eval is not None and holdout.get("auc") is not None else None,
        "final_md_written": (workdir / "FINAL.md").exists(),
        "violations": protocol_violations(workdir, dataset),
        "scored_on": prev.get("scored_on") if prev else (f"{os.uname().sysname}-{os.uname().machine}" + ("-docker" if os.environ.get("BENCH_IN_DOCKER") else "")),
        "provider_error": provider_error(run_dir / "logs") or (
            f"endpoint refusing requests at cell end: {meta['endpoint_at_end']}" if meta.get("endpoint_at_end") not in (None, "ok") else None),
        "usage": usage,
        **cpu_summary(run_dir),
        "cpu_budget_seconds": meta.get("cpu_budget_seconds"),
        "cpu_kill_seconds": meta.get("cpu_kill_seconds"),
        "cpu_seconds_container": meta.get("cpu_seconds_container"),
        "cpu_killed": meta.get("cpu_killed"),
        "commits": commits,
    }
    if result.get("cpu_seconds_python") is not None:
        # the budget applies to counted + uncounted work; validate.sh (exempt) is the contract check, not search
        spent = (result.get("cpu_seconds_counted") or 0) + (result.get("cpu_seconds_uncounted") or 0)
        result["cpu_seconds_budgeted"] = round(spent, 1)
        if meta.get("cpu_budget_seconds"):
            result["cpu_over_budget"] = spent > meta["cpu_budget_seconds"]
        if meta.get("cpu_seconds_container") is not None:
            # container minus Python = the agent process itself (node/python of the harness), plus shell/git
            result["cpu_seconds_overhead"] = round(meta["cpu_seconds_container"] - result["cpu_seconds_python"], 1)
    if best_commit:
        # Score the best-eval *committed* experiment as well as HEAD ("what the agent found" vs "what it delivered").
        # Rows logged on a dirty tree ("<hash>+dirty") are not attributable to a commit and are excluded.
        cand = [e for e in ok if e.get("commit") and not str(e["commit"]).endswith("+dirty")]
        result["best_commit_dirty_rows_excluded"] = len(ok) - len(cand)
        bc = max(cand, key=lambda e: e["eval_auc"])["commit"] if cand else None
        bres = {"best_commit": bc, "best_commit_eval_auc": None, "best_commit_holdout_auc": None,
                "best_commit_holdout_ap": None, "best_commit_error": None, "finalized_best": None}
        if bc:
            bres["best_commit_eval_auc"] = max(e["eval_auc"] for e in cand)
            try:
                same = subprocess.run(["git", "diff", "--quiet", bc, "HEAD", "--", "train.py"], cwd=workdir,
                                      capture_output=True, env={**GIT_ENV, "PATH": os.environ["PATH"]}).returncode == 0
            except Exception:
                same = False
            bres["finalized_best"] = same
            if same and holdout.get("auc") is not None:          # HEAD already is the best commit: reuse its score
                bres["best_commit_holdout_auc"], bres["best_commit_holdout_ap"] = holdout["auc"], holdout.get("ap")
            else:
                sb = score(workdir, dataset, cfg, bc)
                bres["best_commit_holdout_auc"], bres["best_commit_holdout_ap"] = sb.get("auc"), sb.get("ap")
                bres["best_commit_error"] = sb.get("error")
        result.update(bres)
    if all_commits:
        traj = []
        for c in commits[1:]:
            s = score(workdir, dataset, cfg, c["commit"])
            traj.append({**c, "holdout_auc": s.get("auc"), "error": s.get("error")})
        result["trajectory"] = traj
    (run_dir / "eval.json").write_text(json.dumps(result, indent=2))
    return result
