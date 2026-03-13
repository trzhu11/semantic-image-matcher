#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "${LOG_DIR}"

HOST="${QWEN_BACKEND_HOST:-0.0.0.0}"
PORT="${QWEN_BACKEND_PORT:-18080}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
RUN_MODE="${RUN_MODE:-background}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-${ROOT_DIR}/vendor/llama.cpp}"
DEFAULT_SERVER_BIN="${LLAMA_CPP_DIR}/llama-server"
if [[ ! -x "${DEFAULT_SERVER_BIN}" && -x "${LLAMA_CPP_DIR}/build/bin/llama-server" ]]; then
  DEFAULT_SERVER_BIN="${LLAMA_CPP_DIR}/build/bin/llama-server"
fi
LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-${DEFAULT_SERVER_BIN}}"

MODEL_PATH="${QWEN_GGUF_MODEL_PATH:-/share/shared_weights/Qwen3-VL-8B-Instruct-GGUF/Qwen3VL-8B-Instruct-Q8_0.gguf}"
MMPROJ_PATH="${QWEN_GGUF_MMPROJ_PATH:-/share/shared_weights/Qwen3-VL-8B-Instruct-GGUF/mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf}"
MODEL_ALIAS="${QWEN_BACKEND_MODEL:-qwen3-vl-8b-instruct-q8_0}"
CTX_SIZE="${QWEN_CTX_SIZE:-4096}"
GPU_LAYERS="${QWEN_GPU_LAYERS:-all}"
THREADS="${QWEN_THREADS:-8}"
PARALLEL="${QWEN_PARALLEL:-1}"
BATCH_SIZE="${QWEN_BATCH_SIZE:-256}"
UBATCH_SIZE="${QWEN_UBATCH_SIZE:-128}"
THREADS_HTTP="${QWEN_THREADS_HTTP:-4}"
FLASH_ATTN="${QWEN_FLASH_ATTN:-on}"
CACHE_TYPE_K="${QWEN_CACHE_TYPE_K:-q8_0}"
CACHE_TYPE_V="${QWEN_CACHE_TYPE_V:-q8_0}"
REASONING_BUDGET="${QWEN_REASONING_BUDGET:-0}"

if [[ ! -x "${LLAMA_SERVER_BIN}" ]]; then
  echo "llama-server not found: ${LLAMA_SERVER_BIN}" >&2
  echo "run ./scripts/bootstrap_llama_cpp.sh first" >&2
  exit 1
fi

if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "model not found: ${MODEL_PATH}" >&2
  exit 1
fi

if [[ ! -f "${MMPROJ_PATH}" ]]; then
  echo "mmproj not found: ${MMPROJ_PATH}" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${CUDA_DEVICE}}"

COMMAND=(
  "${LLAMA_SERVER_BIN}"
  --host "${HOST}"
  --port "${PORT}"
  --model "${MODEL_PATH}"
  --mmproj "${MMPROJ_PATH}"
  --alias "${MODEL_ALIAS}"
  --ctx-size "${CTX_SIZE}"
  --threads "${THREADS}"
  --threads-http "${THREADS_HTTP}"
  --parallel "${PARALLEL}"
  --n-gpu-layers "${GPU_LAYERS}"
  --split-mode none
  --main-gpu 0
  --batch-size "${BATCH_SIZE}"
  --ubatch-size "${UBATCH_SIZE}"
  --flash-attn "${FLASH_ATTN}"
  --cache-type-k "${CACHE_TYPE_K}"
  --cache-type-v "${CACHE_TYPE_V}"
  --reasoning-budget "${REASONING_BUDGET}"
  --no-webui
)

if [[ "${RUN_MODE}" == "foreground" ]]; then
  exec "${COMMAND[@]}" >> "${LOG_DIR}/llama_backend.log" 2>&1
fi

nohup "${COMMAND[@]}" > "${LOG_DIR}/llama_backend.log" 2>&1 &

echo "Qwen backend started on ${HOST}:${PORT}"
echo "log: ${LOG_DIR}/llama_backend.log"
