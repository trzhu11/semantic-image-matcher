#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
. "${ROOT_DIR}/scripts/load_env.sh"
load_local_env

BACKEND_PORT="${QWEN_BACKEND_PORT:-18080}"
SERVICE_PORT="${SERVICE_PORT:-10003}"
BACKEND_URL="${QWEN_BACKEND_BASE_URL:-http://127.0.0.1:${BACKEND_PORT}}"
SERVICE_URL="${SERVICE_BASE_URL:-http://127.0.0.1:${SERVICE_PORT}}"
BACKEND_SESSION="${BACKEND_SESSION:-qwen_int8_backend}"
SERVICE_SESSION="${SERVICE_SESSION:-qwen_int8_wrapper}"
TMUX_SOCKET_NAME="${TMUX_SOCKET_NAME:-semantic_image_matcher}"

echo "tmux sessions:"
tmux -L "${TMUX_SOCKET_NAME}" has-session -t "${BACKEND_SESSION}" >/dev/null 2>&1 && echo "  ${BACKEND_SESSION}: running" || echo "  ${BACKEND_SESSION}: missing"
tmux -L "${TMUX_SOCKET_NAME}" has-session -t "${SERVICE_SESSION}" >/dev/null 2>&1 && echo "  ${SERVICE_SESSION}: running" || echo "  ${SERVICE_SESSION}: missing"
echo "  socket: ${TMUX_SOCKET_NAME}"

echo
echo "ports:"
ss -ltnp | rg "(:${BACKEND_PORT}|:${SERVICE_PORT})" || true

echo
echo "health:"
curl --noproxy '*' -sf "${BACKEND_URL}/v1/models" >/dev/null && echo "  backend ok: ${BACKEND_URL}/v1/models" || echo "  backend failed: ${BACKEND_URL}/v1/models"
curl --noproxy '*' -sf "${SERVICE_URL}/health" >/dev/null && echo "  wrapper ok: ${SERVICE_URL}/health" || echo "  wrapper failed: ${SERVICE_URL}/health"
