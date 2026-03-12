#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_PORT="${QWEN_BACKEND_PORT:-18080}"
SERVICE_PORT="${SERVICE_PORT:-10003}"
BACKEND_URL="${QWEN_BACKEND_BASE_URL:-http://127.0.0.1:${BACKEND_PORT}}"
SERVICE_URL="${SERVICE_BASE_URL:-http://127.0.0.1:${SERVICE_PORT}}"
WAIT_RETRIES="${WAIT_RETRIES:-120}"
WAIT_INTERVAL_SEC="${WAIT_INTERVAL_SEC:-1}"

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

if ! curl --noproxy '*' -sf "${BACKEND_URL}/v1/models" >/dev/null 2>&1; then
  "${ROOT_DIR}/scripts/start_llama_backend.sh"
fi
wait_for_url "Qwen backend" "${BACKEND_URL}/v1/models"

if ! curl --noproxy '*' -sf "${SERVICE_URL}/health" >/dev/null 2>&1; then
  "${ROOT_DIR}/scripts/start_service.sh"
fi
wait_for_url "Wrapper service" "${SERVICE_URL}/health"

echo "Replacement stack is ready."
echo "Qwen backend: ${BACKEND_URL}"
echo "Wrapper service: ${SERVICE_URL}"
