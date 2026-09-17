#!/usr/bin/env bash
# Claude Code -> Ollama (Anthropic Messages API compatibility at $BENCH_LLM_BASE_URL/v1/messages)
source "$(dirname "$0")/_lib.sh"
export CLAUDE_CONFIG_DIR="$BENCH_STATE_DIR/claude"          # fresh config: no personal hooks/plugins/MCP/CLAUDE.md
mkdir -p "$CLAUDE_CONFIG_DIR"
export ANTHROPIC_BASE_URL="$BENCH_LLM_BASE_URL"
export ANTHROPIC_AUTH_TOKEN="$BENCH_LLM_API_KEY"            # sent as Authorization: Bearer (what ollama.com expects)
export ANTHROPIC_API_KEY=""
export DISABLE_TELEMETRY=1 DISABLE_AUTOUPDATER=1 CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
export CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1             # strip beta headers a non-Anthropic backend may reject
export CLAUDE_CODE_ATTRIBUTION_HEADER=0                     # Anthropic's own advice for gateways: the attribution block
                                                            # is prepended as prompt text and breaks prefix caching
# Every model Claude Code might route to (background summaries/titles, subagents) is the benchmark model, so the
# cell uses exactly one model and no request 404s against Ollama.
export ANTHROPIC_DEFAULT_HAIKU_MODEL="$BENCH_MODEL" ANTHROPIC_DEFAULT_SONNET_MODEL="$BENCH_MODEL" ANTHROPIC_DEFAULT_OPUS_MODEL="$BENCH_MODEL"
export CLAUDE_CODE_SUBAGENT_MODEL="$BENCH_MODEL"
export API_TIMEOUT_MS=600000
log "claude --model $BENCH_MODEL via $ANTHROPIC_BASE_URL"
# Session persistence ON since phase 2: the per-request transcript lands under $CLAUDE_CONFIG_DIR/projects/ (inside
# the run dir, pulled with the cell), so Claude Code cells can be audited like the JSONL-emitting harnesses.
exec claude -p "$PROMPT" \
  --model "$BENCH_MODEL" \
  --dangerously-skip-permissions \
  --output-format json
