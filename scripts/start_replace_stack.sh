#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_PORT="${QWEN_BACKEND_PORT:-18080}"
SERVICE_PORT="${SERVICE_PORT:-10003}"
BACKEND_URL="${QWEN_BACKEND_BASE_URL:-http://127.0.0.1:${BACKEND_PORT}}"
SERVICE_URL="${SERVICE_BASE_URL:-http://127.0.0.1:${SERVICE_PORT}}"
WAIT_RETRIES="${WAIT_RETRIES:-120}"
WAIT_INTERVAL_SEC="${WAIT_INTERVAL_SEC:-1}"
PROCESS_MANAGER="${PROCESS_MANAGER:-tmux}"
BACKEND_SESSION="${BACKEND_SESSION:-qwen_int8_backend}"
SERVICE_SESSION="${SERVICE_SESSION:-qwen_int8_wrapper}"
TMUX_SOCKET_NAME="${TMUX_SOCKET_NAME:-semantic_image_matcher}"

wait_for_url() {
  local name="$1"
  local url="$2"
  local attempt

  for ((attempt = 1; attempt <= WAIT_RETRIES; attempt++)); do
    if curl --noproxy '*' -sf "${url}" >/dev/null; then
      echo "${name} is ready: ${url}"
      return 0
    fi
    sleep "${WAIT_INTERVAL_SEC}"
  done

  echo "${name} failed to become ready: ${url}" >&2
  return 1
}

start_tmux_session() {
  local session_name="$1"
  local command="$2"

  if tmux -L "${TMUX_SOCKET_NAME}" has-session -t "${session_name}" >/dev/null 2>&1; then
    echo "tmux session already exists: ${session_name}"
    return 0
  fi

  tmux -L "${TMUX_SOCKET_NAME}" new-session -d -s "${session_name}" "${command}"
}

if ! curl --noproxy '*' -sf "${BACKEND_URL}/v1/models" >/dev/null 2>&1; then
  if [[ "${PROCESS_MANAGER}" == "tmux" ]] && command -v tmux >/dev/null 2>&1; then
    start_tmux_session \
      "${BACKEND_SESSION}" \
      "cd '${ROOT_DIR}' && CUDA_DEVICE='${CUDA_DEVICE:-0}' RUN_MODE=foreground ./scripts/start_llama_backend.sh"
  else
    "${ROOT_DIR}/scripts/start_llama_backend.sh"
  fi
fi
wait_for_url "Qwen backend" "${BACKEND_URL}/v1/models"

if ! curl --noproxy '*' -sf "${SERVICE_URL}/health" >/dev/null 2>&1; then
  if [[ "${PROCESS_MANAGER}" == "tmux" ]] && command -v tmux >/dev/null 2>&1; then
    start_tmux_session \
      "${SERVICE_SESSION}" \
      "cd '${ROOT_DIR}' && CUDA_DEVICE='${CUDA_DEVICE:-0}' RUN_MODE=foreground ./scripts/start_service.sh"
  else
    "${ROOT_DIR}/scripts/start_service.sh"
  fi
fi
wait_for_url "Wrapper service" "${SERVICE_URL}/health"

echo "Replacement stack is ready."
echo "Qwen backend: ${BACKEND_URL}"
echo "Wrapper service: ${SERVICE_URL}"
if [[ "${PROCESS_MANAGER}" == "tmux" ]] && command -v tmux >/dev/null 2>&1; then
  echo "tmux backend session: ${BACKEND_SESSION}"
  echo "tmux wrapper session: ${SERVICE_SESSION}"
  echo "tmux socket: ${TMUX_SOCKET_NAME}"
fi
