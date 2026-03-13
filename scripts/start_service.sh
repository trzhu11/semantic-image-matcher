#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "${LOG_DIR}"

CONDA_PYTHON="${CONDA_PYTHON:-/home/test/.conda/envs/semantic_matcher/bin/python}"
HOST="${SERVICE_HOST:-0.0.0.0}"
PORT="${SERVICE_PORT:-10003}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
RUN_MODE="${RUN_MODE:-background}"

export QWEN_BACKEND_BASE_URL="${QWEN_BACKEND_BASE_URL:-http://127.0.0.1:18080}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${CUDA_DEVICE}}"
export DINO_DEVICE="${DINO_DEVICE:-cuda:0}"
export QWEN_MAX_NEW_TOKENS="${QWEN_MAX_NEW_TOKENS:-4}"
export MAX_IMAGE_DIM="${MAX_IMAGE_DIM:-768}"
export RERANK_TOP_K="${RERANK_TOP_K:-1}"
export RERANK_EXECUTION="${RERANK_EXECUTION:-sequential}"

COMMAND=(
  "${CONDA_PYTHON}"
  -m
  uvicorn
  qwen3_vl_single_gpu.app:app
  --app-dir
  "${ROOT_DIR}/src"
  --host
  "${HOST}"
  --port
  "${PORT}"
)

if [[ "${RUN_MODE}" == "foreground" ]]; then
  exec "${COMMAND[@]}" >> "${LOG_DIR}/wrapper_service.log" 2>&1
fi

nohup "${COMMAND[@]}" > "${LOG_DIR}/wrapper_service.log" 2>&1 &

echo "wrapper service started on ${HOST}:${PORT}"
echo "log: ${LOG_DIR}/wrapper_service.log"
