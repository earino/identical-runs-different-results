#!/usr/bin/env bash
# OpenCode -> Ollama via OpenAI-compatible provider at $OPENAI_COMPAT_URL
source "$(dirname "$0")/_lib.sh"
export XDG_CONFIG_HOME="$BENCH_STATE_DIR/xdg"                # hide the operator's ~/.config/opencode (plugins, providers)
export XDG_DATA_HOME="$BENCH_STATE_DIR/xdg-data"             # sessions, snapshots, opencode.db: per run, not ~/.local/share
export XDG_STATE_HOME="$BENCH_STATE_DIR/xdg-state"
export XDG_CACHE_HOME="$BENCH_STATE_DIR/xdg-cache"
export OPENCODE_CONFIG_DIR="$BENCH_STATE_DIR/opencode"
export OPENCODE_CONFIG="$BENCH_STATE_DIR/opencode/opencode.json"
mkdir -p "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" "$XDG_STATE_HOME" "$XDG_CACHE_HOME" "$OPENCODE_CONFIG_DIR"
export BENCH_LLM_API_KEY
cat > "$OPENCODE_CONFIG" <<JSON
{
  "\$schema": "https://opencode.ai/config.json",
  "model": "bench/$BENCH_MODEL",
  "enabled_providers": ["bench"],
  "permission": "allow",
  "share": "disabled",
  "provider": {
    "bench": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Ollama (bench)",
      "options": {"baseURL": "$OPENAI_COMPAT_URL", "apiKey": "{env:BENCH_LLM_API_KEY}"},
      "models": {"$BENCH_MODEL": {"name": "$BENCH_MODEL", "limit": {"context": 128000, "output": 32000}}}
    }
  }
}
JSON
log "opencode run -m bench/$BENCH_MODEL via $OPENAI_COMPAT_URL"
exec opencode run -m "bench/$BENCH_MODEL" --format json --dangerously-skip-permissions --pure "$PROMPT"
