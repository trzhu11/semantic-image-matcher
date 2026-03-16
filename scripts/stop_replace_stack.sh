#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
. "${ROOT_DIR}/scripts/load_env.sh"
load_local_env

BACKEND_SESSION="${BACKEND_SESSION:-qwen_int8_backend}"
SERVICE_SESSION="${SERVICE_SESSION:-qwen_int8_wrapper}"
TMUX_SOCKET_NAME="${TMUX_SOCKET_NAME:-semantic_image_matcher}"
BACKEND_PORT="${QWEN_BACKEND_PORT:-18080}"
SERVICE_PORT="${SERVICE_PORT:-10003}"

terminate_process_on_port() {
  local port="$1"
  local pid
  pid="$(lsof -t -iTCP:${port} -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -n "${pid}" ]]; then
    kill "${pid}"
    echo "killed process on port ${port}: ${pid}"
  else
    echo "no listening process on port ${port}"
  fi
}

if tmux -L "${TMUX_SOCKET_NAME}" has-session -t "${SERVICE_SESSION}" >/dev/null 2>&1; then
  tmux -L "${TMUX_SOCKET_NAME}" kill-session -t "${SERVICE_SESSION}"
  echo "stopped tmux session: ${SERVICE_SESSION}"
else
  echo "tmux session not found: ${SERVICE_SESSION}"
fi

if tmux -L "${TMUX_SOCKET_NAME}" has-session -t "${BACKEND_SESSION}" >/dev/null 2>&1; then
  tmux -L "${TMUX_SOCKET_NAME}" kill-session -t "${BACKEND_SESSION}"
  echo "stopped tmux session: ${BACKEND_SESSION}"
else
  echo "tmux session not found: ${BACKEND_SESSION}"
fi

terminate_process_on_port "${SERVICE_PORT}"
terminate_process_on_port "${BACKEND_PORT}"
