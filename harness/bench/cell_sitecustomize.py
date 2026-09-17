"""Cell-side CPU ledger and XGBoost fit log (phase 2 budget).

Installed as `sitecustomize.py` in the cell image's venv, so EVERY Python process that uses /app/.venv/bin/python
imports it at start: train.py runs under run_experiment.sh, the agent's own scripts, one-liners. It is a no-op
outside a cell (no /cell/logs and no BENCH_LEDGER_DIR), so the orchestrator and the evaluator are unaffected.

What it does, per process:
  start   reads logs/cpu_ledger.tsv; if the cell's Python CPU seconds already exceed BENCH_CPU_BUDGET_S and this
          process is not exempt (validate.sh), it refuses to run (exit 3) and logs the refusal
  run     a watchdog thread stops the process (exit 3) if the budget is crossed mid-run, so a long sweep cannot
          overshoot; every xgboost train/cv call is appended to logs/fits.tsv with rows and seconds
  exit    appends one ledger row: CPU seconds (self + waited children, user + sys), kind (counted = launched by
          run_experiment.sh, uncounted = anything else, exempt = validate.sh), event (exit | term | refused |
          over-budget), wall seconds, fit count, argv. SIGTERM (the wrapper's timeout) is recorded too; SIGKILL
          (OOM) is not, which is why the container's own CPU accounting is kept alongside (bench/runner.py).

Budget values come from the environment, else from <run dir>/bench.env, because some harnesses run commands in a
clean environment. Known bypass: `python -S` skips site; the container CPU backstop and the audit cover it.
"""
import atexit
import os
import resource
import signal
import sys
import threading
import time


def _ledger_dir():
    d = os.environ.get("BENCH_LEDGER_DIR")
    if d and os.path.isdir(d):
        return d
    if os.path.isdir("/cell/logs"):
        return "/cell/logs"
    return None


_DIR = _ledger_dir()
if _DIR:
    _LEDGER = os.path.join(_DIR, "cpu_ledger.tsv")
    _FITS = os.path.join(_DIR, "fits.tsv")
    _LEDGER_HEADER = "ts\tpid\tcpu_s\tkind\tevent\twall_s\tfits\targv\n"
    _FITS_HEADER = "ts\tpid\tkind\tapi\trows\tseconds\n"
    _fileenv = {}
    try:
        with open(os.path.join(os.path.dirname(_DIR.rstrip("/")), "bench.env")) as _f:
            for _line in _f:
                if "=" in _line:
                    _k, _v = _line.rstrip("\n").split("=", 1)
                    _fileenv[_k] = _v
    except OSError:
        pass

    def _get(k, default=""):
        return os.environ.get(k) or _fileenv.get(k) or default

    try:
        _BUDGET = float(_get("BENCH_CPU_BUDGET_S", "0") or 0)
    except ValueError:
        _BUDGET = 0.0
    _EXEMPT = os.environ.get("BENCH_LEDGER_EXEMPT") == "1"
    _KIND = "exempt" if _EXEMPT else ("counted" if os.environ.get("BENCH_COUNTED") == "1" else "uncounted")
    _ARGV = " ".join(sys.argv[:4])[:160].replace("\t", " ").replace("\n", " ")
    _T0 = time.time()
    _state = {"fits": 0, "done": False}

    def _append(path, header, row):
        new = not os.path.exists(path)
        with open(path, "a") as f:
            if new:
                f.write(header)
            f.write(row)

    def _used_before():
        total = 0.0
        try:
            with open(_LEDGER) as f:
                for line in f:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) >= 3 and parts[0] != "ts":
                        try:
                            total += float(parts[2])
                        except ValueError:
                            pass
        except OSError:
            pass
        return total

    def _self_cpu():
        a = resource.getrusage(resource.RUSAGE_SELF)
        b = resource.getrusage(resource.RUSAGE_CHILDREN)
        return a.ru_utime + a.ru_stime + b.ru_utime + b.ru_stime

    def _record(event="exit"):
        if _state["done"]:
            return
        _state["done"] = True
        try:
            _append(_LEDGER, _LEDGER_HEADER,
                    f"{int(time.time())}\t{os.getpid()}\t{_self_cpu():.3f}\t{_KIND}\t{event}\t{time.time() - _T0:.1f}\t{_state['fits']}\t{_ARGV}\n")
        except OSError:
            pass

    _USED0 = _used_before()
    if _BUDGET and not _EXEMPT and _USED0 >= _BUDGET:
        sys.stderr.write(f"CPU BUDGET EXHAUSTED: {_USED0:.0f} of {_BUDGET:.0f} CPU seconds of Python compute used in this cell. "
                         "No more Python runs. Finalize now (see program.md).\n")
        try:
            _append(_LEDGER, _LEDGER_HEADER, f"{int(time.time())}\t{os.getpid()}\t0.000\t{_KIND}\trefused\t0.0\t0\t{_ARGV}\n")
        except OSError:
            pass
        os._exit(3)

    atexit.register(_record)

    def _on_term(signum, frame):
        _record("term")
        os._exit(143)

    try:
        signal.signal(signal.SIGTERM, _on_term)
    except (ValueError, OSError):
        pass

    if _BUDGET and not _EXEMPT:
        def _watch():
            while True:
                time.sleep(10)
                if _USED0 + _self_cpu() >= _BUDGET:
                    sys.stderr.write(f"CPU BUDGET EXHAUSTED mid-run: {_USED0 + _self_cpu():.0f} of {_BUDGET:.0f} CPU seconds. "
                                     "Stopping this process. Finalize now (see program.md).\n")
                    sys.stderr.flush()
                    _record("over-budget")
                    os._exit(3)
        threading.Thread(target=_watch, name="bench-cpu-watchdog", daemon=True).start()

    # --- XGBoost fit log: wrap train/cv at every binding user code and the sklearn wrappers call -----------------
    import importlib.abc
    import importlib.machinery

    def _rows(a, k):
        d = a[1] if len(a) > 1 else k.get("dtrain")
        try:
            return d.num_row()
        except Exception:
            return ""

    def _wrap(fn, api):
        def wrapped(*a, **k):
            t = time.time()
            try:
                return fn(*a, **k)
            finally:
                _state["fits"] += 1
                try:
                    _append(_FITS, _FITS_HEADER, f"{int(time.time())}\t{os.getpid()}\t{_KIND}\t{api}\t{_rows(a, k)}\t{time.time() - t:.2f}\n")
                except OSError:
                    pass
        wrapped.__wrapped__ = fn
        return wrapped

    def _patch(xgb):
        try:
            xgb.train = _wrap(xgb.train, "train")
            xgb.cv = _wrap(xgb.cv, "cv")
            xgb.training.train = _wrap(xgb.training.train, "train")
            xgb.sklearn.train = _wrap(xgb.sklearn.train, "train")     # XGBClassifier/XGBRegressor.fit call this name
        except Exception:
            pass

    class _XGBFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name != "xgboost":
                return None
            try:
                sys.meta_path.remove(self)
            except ValueError:
                pass
            spec = importlib.machinery.PathFinder.find_spec(name, path)
            if spec is None or spec.loader is None:
                return None
            loader = spec.loader
            orig = loader.exec_module

            def exec_module(module):
                orig(module)
                _patch(module)
            loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _XGBFinder())
