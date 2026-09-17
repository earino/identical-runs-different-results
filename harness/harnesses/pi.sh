#!/usr/bin/env bash
# pi (earendil-works/pi coding agent) -> Ollama via OpenAI chat-completions at $OPENAI_COMPAT_URL
source "$(dirname "$0")/_lib.sh"
export PI_CODING_AGENT_DIR="$BENCH_STATE_DIR/pi"            # fresh agent dir: our models.json only
mkdir -p "$PI_CODING_AGENT_DIR"
export BENCH_LLM_API_KEY
cat > "$PI_CODING_AGENT_DIR/models.json" <<JSON
{
  "providers": {
    "bench": {
      "baseUrl": "$OPENAI_COMPAT_URL",
      "api": "openai-completions",
      "apiKey": "\$BENCH_LLM_API_KEY",
      "compat": {"supportsDeveloperRole": false, "supportsReasoningEffort": false},
      "models": [{"id": "$BENCH_MODEL", "name": "$BENCH_MODEL", "contextWindow": 128000, "maxTokens": 16384}]
    }
  }
}
JSON
cat > "$PI_CODING_AGENT_DIR/settings.json" <<JSON
{"defaultProvider": "bench", "defaultModel": "$BENCH_MODEL", "quietStartup": true}
JSON
log "pi --model bench/$BENCH_MODEL via $OPENAI_COMPAT_URL"
# --mode json: non-interactive, emits JSON events incl. per-message usage. Context files (AGENTS.md) stay enabled.
exec pi --mode json --no-session --no-extensions --no-skills --no-prompt-templates \
  --provider bench --model "$BENCH_MODEL" "$PROMPT"
