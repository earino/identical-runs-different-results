#!/usr/bin/env bash
# OpenClaw -> Ollama via its native `ollama` provider (/api/chat; do NOT add /v1 — tool calling is unreliable there),
# or -> an OpenAI-compatible gateway (BENCH_MODEL_ENDPOINT other than cloud/local, e.g. lunaroute) via `openai-completions`.
# Install: npm install -g openclaw@latest --allow-scripts=openclaw   (Node >= 24.16)
source "$(dirname "$0")/_lib.sh"
command -v openclaw >/dev/null || { log "openclaw not installed"; exit 127; }
export OPENCLAW_STATE_DIR="$BENCH_STATE_DIR/openclaw"       # sessions/memory per run
mkdir -p "$OPENCLAW_STATE_DIR"
# Phase 2: `agent exec` keeps its session state (the per-command transcript) ONLY with an explicit --state-dir;
# without it the state is ephemeral and deleted after the run, which is why phase-1 OpenClaw cells had no transcript.
CFG="$BENCH_STATE_DIR/openclaw.json5"
export OLLAMA_API_KEY="$BENCH_LLM_API_KEY"
case "${BENCH_MODEL_ENDPOINT:-cloud}" in
  cloud|local) PROVIDER=ollama; API=ollama;             BASE="$BENCH_LLM_BASE_URL" ;;
  *)           PROVIDER=bench;  API=openai-completions; BASE="$OPENAI_COMPAT_URL" ;;
esac
cat > "$CFG" <<JSON5
{
  models: { providers: { $PROVIDER: { baseUrl: "$BASE", apiKey: "OLLAMA_API_KEY", api: "$API",
    models: [{ id: "$BENCH_MODEL", name: "$BENCH_MODEL", contextWindow: 128000, maxTokens: 16384 }] } } },
  agents: { defaults: { model: { primary: "$PROVIDER/$BENCH_MODEL" }, timeoutSeconds: $WALL } },
  tools: { exec: { mode: "full" } }
}
JSON5
log "openclaw agent exec --model $PROVIDER/$BENCH_MODEL via $BASE ($API)"
exec openclaw agent exec --config "$CFG" --state-dir "$OPENCLAW_STATE_DIR" --message-file "$BENCH_PROMPT_FILE" --cwd "$BENCH_WORKDIR" \
  --model "$PROVIDER/$BENCH_MODEL" --timeout "$WALL" --json
