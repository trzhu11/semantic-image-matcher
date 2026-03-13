#!/usr/bin/env bash
set -euo pipefail

BACKEND_SESSION="${BACKEND_SESSION:-qwen_int8_backend}"
SERVICE_SESSION="${SERVICE_SESSION:-qwen_int8_wrapper}"
TMUX_SOCKET_NAME="${TMUX_SOCKET_NAME:-semantic_image_matcher}"

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
