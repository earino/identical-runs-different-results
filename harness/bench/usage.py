"""Best-effort token-usage extraction from harness stdout logs.

Formats differ per harness (and drift over time), so this walks every JSON object in the log and picks up
usage-shaped dicts. Input tokens are reported as uncached + cached so harnesses with heavy prompt caching
(all of them) are comparable on total prompt volume.

  claude   one final `result` object with cumulative usage{input_tokens, cache_read_input_tokens, output_tokens}
  pi       per-message `message_end` events: message.usage{input, output, cacheRead, cacheWrite}
  opencode per-step `step_finish` events: part.tokens{input, output, reasoning, cache{read, write}}
  codex    `turn.completed` events: usage{input_tokens, cached_input_tokens, output_tokens, reasoning_output_tokens}
  hermes   --usage-file JSON (input_tokens, cache_read_tokens, output_tokens, api_calls) if present
  openclaw final envelope usage{input, output, total}
"""
from __future__ import annotations

import json
from pathlib import Path

IN_KEYS = ("input_tokens", "prompt_tokens", "input", "prompt", "inputTokens")
OUT_KEYS = ("output_tokens", "completion_tokens", "output", "completion", "outputTokens")
CACHED_KEYS = ("cache_read_input_tokens", "cached_input_tokens", "cacheRead", "cache_read_tokens", "cacheReadInputTokens", "cached_tokens")
INCLUSIVE_INPUT = {"codex"}   # OpenAI semantics: input_tokens already includes cached_input_tokens
CUMULATIVE = {"claude", "openclaw", "hermes"}   # one final object with totals (vs per-event deltas to sum)
# per-harness: only these top-level event types carry authoritative usage (others are partial/streaming updates)
EVENT_FILTER = {"pi": {"message_end"}, "opencode": {"step_finish"}, "codex": {"turn.completed"}}


def _json_objects(text: str):
    text = text.strip()
    if not text:
        return
    try:
        yield json.loads(text)
        return
    except Exception:
        pass
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                yield json.loads(line)
            except Exception:
                continue


def _walk(obj, path=()):
    if isinstance(obj, dict):
        yield path, obj
        for k, v in obj.items():
            yield from _walk(v, path + (k,))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk(v, path + (i,))


def _num(d: dict, keys) -> int | None:
    for k in keys:
        v = d.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return int(v)
    return None


def _usage_dicts(obj):
    for path, d in _walk(obj):
        tail = path[-1] if path else None
        looks_like = tail in ("usage", "tokens", "token_usage") or path == () and "input_tokens" in d or (
            _num(d, IN_KEYS) is not None and _num(d, OUT_KEYS) is not None and tail != "cost")
        if not looks_like:
            continue
        ti, to = _num(d, IN_KEYS), _num(d, OUT_KEYS)
        if ti is None and to is None:
            continue
        cached = _num(d, CACHED_KEYS)
        if cached is None and isinstance(d.get("cache"), dict):       # opencode: tokens.cache.read
            cached = _num(d["cache"], ("read",))
        yield ti or 0, to or 0, cached or 0


def hermes_usage_from_db(run_dir: Path) -> dict | None:
    """Hermes keeps per-session token totals in <HERMES_HOME>/state.db; the per-run HERMES_HOME holds exactly one
    session. Read it directly (no dependency on the 'session_id:' stdout line or the sqlite3 CLI)."""
    import sqlite3
    db = run_dir / "harness_state" / "hermes" / "state.db"
    if not db.exists():
        return None
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        row = con.execute("select input_tokens, output_tokens, cache_read_tokens, api_call_count, message_count "
                          "from sessions order by started_at desc limit 1").fetchone()
        con.close()
    except Exception:
        return None
    if not row:
        return None
    ti, to, cached, calls, msgs = (x or 0 for x in row)
    return {"input_tokens": ti, "output_tokens": to, "cache_read_tokens": cached, "api_calls": calls, "message_count": msgs}


def extract_usage(harness: str, log_dir: Path) -> dict:
    out = {"tokens_in": None, "tokens_in_uncached": None, "tokens_in_cached": None, "tokens_out": None,
           "turns": None, "api_calls": None, "method": "none"}
    texts = []
    for name in ("harness.stdout", "usage.json"):
        p = log_dir / name
        if p.exists():
            texts.append(p.read_text(errors="replace"))
    if harness == "hermes" and not (log_dir / "usage.json").exists():
        db_usage = hermes_usage_from_db(log_dir.parent)
        if db_usage:
            texts.append(json.dumps({"usage": db_usage, "api_calls": db_usage["api_calls"]}))
    if not texts:
        return out
    events = []
    turns = api_calls = None
    allowed = EVENT_FILTER.get(harness)
    for text in texts:
        for obj in _json_objects(text):
            if allowed and isinstance(obj, dict) and obj.get("type") not in allowed:
                continue
            events.extend(_usage_dicts(obj))
            for _, d in _walk(obj):
                if isinstance(d.get("num_turns"), int):
                    turns = d["num_turns"]
                if isinstance(d.get("assistantTurns"), int):
                    turns = d["assistantTurns"]
                if isinstance(d.get("api_calls"), int):
                    api_calls = d["api_calls"]
    if not events:
        return out
    if harness in CUMULATIVE:
        ti, to, cached = max(events, key=lambda e: e[0] + e[2])
        method = "cumulative-max"
    else:
        ti, to, cached = (sum(e[i] for e in events) for i in range(3))
        method = f"sum-of-{len(events)}-events"
        api_calls = api_calls or len(events)
    if harness in INCLUSIVE_INPUT:
        ti = ti - cached   # split the inclusive figure into uncached + cached like the other harnesses
    out.update({"tokens_in": ti + cached, "tokens_in_uncached": ti, "tokens_in_cached": cached, "tokens_out": to,
                "turns": turns, "api_calls": api_calls, "method": method})
    return out
