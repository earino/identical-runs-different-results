#!/usr/bin/env bash
# OpenAI Codex CLI -> Ollama via a custom provider speaking the (non-stateful) Responses API at $OPENAI_COMPAT_URL
source "$(dirname "$0")/_lib.sh"
export CODEX_HOME="$BENCH_STATE_DIR/codex"                  # fresh config/auth dir
mkdir -p "$CODEX_HOME"
export BENCH_LLM_API_KEY
cat > "$CODEX_HOME/config.toml" <<TOML
model = "$BENCH_MODEL"
model_provider = "bench"
model_context_window = 128000
approval_policy = "never"
sandbox_mode = "danger-full-access"
web_search = "disabled"            # ollama.com rejects requests that carry the web_search tool

[model_providers.bench]
name = "Ollama (bench)"
base_url = "$OPENAI_COMPAT_URL"
env_key = "BENCH_LLM_API_KEY"
wire_api = "responses"
TOML
log "codex exec -m $BENCH_MODEL via $OPENAI_COMPAT_URL"
exec codex exec --json --skip-git-repo-check --ephemeral -C "$BENCH_WORKDIR" \
  --dangerously-bypass-approvals-and-sandbox \
  -m "$BENCH_MODEL" -o "$BENCH_LOG_DIR/last_message.txt" - < "$BENCH_PROMPT_FILE"
