from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from bench import ROOT

LOCAL_OLLAMA = "http://localhost:11434"


@dataclass
class Model:
    name: str                 # exact model id at the provider, e.g. "kimi-k3" (Ollama Cloud) or "glm-5.3" (LunaRoute)
    endpoint: str = "cloud"   # "cloud" -> ollama.com with API key; "local" -> local `ollama serve`, no key;
                              # any other name -> a provider block of that name in bench.yaml ({base_url, api_key_env}), e.g. "lunaroute"
    label: str | None = None  # display name (defaults to name)

    @property
    def slug(self) -> str:
        return re.sub(r"[^A-Za-z0-9._-]+", "_", self.name).replace(":", "_")

    @property
    def display(self) -> str:
        return self.label or self.name


@dataclass
class Config:
    raw: dict
    path: Path

    # --- accessors ---------------------------------------------------------
    @property
    def harnesses(self) -> list[str]:
        return list(self.raw.get("harnesses", []))

    @property
    def models(self) -> list[Model]:
        out = []
        for m in self.raw.get("models", []):
            if isinstance(m, str):
                out.append(Model(name=m))
            else:
                out.append(Model(name=m["name"], endpoint=m.get("endpoint", "cloud"), label=m.get("label")))
        return out

    def model(self, name: str) -> Model:
        for m in self.models:
            if m.name == name or m.slug == name:
                return m
        # ad-hoc models not in the config: "local:<name>" -> local ollama, "<provider>:<name>" -> that provider block
        # (e.g. "lunaroute:glm-5.3"), anything else -> Ollama Cloud
        if name.startswith("local:"):
            return Model(name=name[len("local:"):], endpoint="local")
        head, sep, rest = name.partition(":")
        if sep and head != "ollama" and isinstance(self.raw.get(head), dict) and "base_url" in self.raw[head]:
            return Model(name=rest, endpoint=head)
        return Model(name=name, endpoint="cloud")

    @property
    def datasets(self) -> list[str]:
        return list(self.raw.get("datasets", []))

    @property
    def seeds(self) -> list[int]:
        return [int(s) for s in self.raw.get("seeds", [1])]

    @property
    def parallel(self) -> int:
        return int(self.raw.get("parallel", 1))

    def price(self, model_name: str) -> dict | None:
        """{input, cached, output} USD per 1M tokens, or None if the model is not in bench.yaml `prices`."""
        p = (self.raw.get("prices") or {}).get(model_name)
        return {k: float(p[k]) for k in ("input", "cached", "output")} if p else None

    @property
    def max_cells_per_hour(self) -> float:
        return float(self.raw.get("max_cells_per_hour", 0) or 0)

    @property
    def isolation(self) -> str:
        return str(self.raw.get("isolation", "container"))

    @property
    def cell_cpus(self) -> str:
        return str(self.raw.get("cell_cpus", self.threads))

    @property
    def cell_memory(self) -> str:
        return str(self.raw.get("cell_memory", "8g"))

    @property
    def research_allowed(self) -> bool:
        return bool(self.raw.get("research_allowed", False))

    @property
    def prompt(self) -> str:
        return str(self.raw.get("prompt", "Read program.md and follow it exactly.")).strip()

    @property
    def wall_clock_minutes(self) -> int:
        """Legacy flat value; prefer wall_clock(dataset)."""
        v = self.raw["budget"].get("wall_clock_minutes", "auto")
        return int(v) if v != "auto" else self.wall_clock("default")

    def wall_clock(self, dataset: str) -> int:
        """Ceiling in minutes for one cell. 'auto' = max_experiments x per-experiment timeout + thinking allowance,
        so the experiment cap is the binding budget and the clock only catches hangs."""
        v = self.raw["budget"].get("wall_clock_minutes", "auto")
        if v != "auto":
            return int(v)
        allowance = int(self.raw["budget"].get("thinking_allowance_minutes", 150))
        return self.max_experiments * self.experiment_timeout(dataset) // 60 + allowance

    @property
    def cell_max_gb(self) -> float:
        """Kill a cell whose run directory grows past this (a runaway tool-output log filled a 150 GB disk once). 0 = off."""
        return float(self.raw["budget"].get("cell_max_gb", 5))

    @property
    def stall_minutes(self) -> int:
        """Kill a cell that has written nothing (no experiment row, no workdir file) for this long. 0 = off."""
        return int(self.raw["budget"].get("stall_minutes", 30))

    @property
    def max_experiments(self) -> int:
        return int(self.raw["budget"]["max_experiments"])

    @property
    def threads(self) -> int:
        return int(self.raw["budget"].get("cpu_threads_per_run", 4))

    def experiment_timeout(self, dataset: str) -> int:
        t = self.raw.get("experiment_timeout_seconds", {}) or {}
        return int(t.get(dataset, t.get("default", 120)))

    def cpu_budget(self, dataset: str) -> int:
        """Phase-2 budget: Python CPU seconds per cell, enforced by the cell-side ledger hook. 0 = off (phase 1)."""
        t = self.raw.get("cpu_budget_seconds", {}) or {}
        return int(t.get(dataset, t.get("default", 0)) or 0)

    def cpu_kill(self, dataset: str) -> int:
        """Hard backstop: container CPU seconds at which the orchestrator kills the cell. 0 = off."""
        t = self.raw.get("cpu_kill_seconds", {}) or {}
        return int(t.get(dataset, t.get("default", 0)) or 0)

    # --- endpoints ---------------------------------------------------------
    def llm_endpoint(self, model: Model) -> tuple[str, str]:
        """(base_url, api_key) the harness should talk to for this model."""
        if model.endpoint == "local":
            return os.environ.get("OLLAMA_LOCAL_URL", LOCAL_OLLAMA), "ollama"
        if model.endpoint == "cloud":
            o = self.raw.get("ollama", {}) or {}
            base = os.environ.get("OLLAMA_BASE_URL") or o.get("base_url", "https://ollama.com")
            key = os.environ.get(o.get("api_key_env", "OLLAMA_API_KEY"), "")
            return base.rstrip("/"), key
        # any other endpoint is an OpenAI-compatible gateway described by a top-level block of that name in bench.yaml
        o = self.raw.get(model.endpoint)
        if not isinstance(o, dict) or "base_url" not in o:
            raise KeyError(f"model {model.name}: no provider block '{model.endpoint}:' with base_url in {self.path.name}")
        key = os.environ.get(o.get("api_key_env", f"{model.endpoint.upper()}_API_KEY"), "")
        return str(o["base_url"]).rstrip("/"), key


def load_config(path: str | Path | None = None) -> Config:
    p = Path(path) if path else ROOT / "bench.yaml"
    with open(p) as f:
        return Config(raw=yaml.safe_load(f) or {}, path=p)
