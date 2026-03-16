#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
. "${ROOT_DIR}/scripts/load_env.sh"
load_local_env

LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "${LOG_DIR}"

CONDA_PYTHON="${CONDA_PYTHON:-/home/test/.conda/envs/semantic_matcher/bin/python}"
HOST="${SERVICE_HOST:-0.0.0.0}"
PORT="${SERVICE_PORT:-10003}"
SERVICE_CUDA_DEVICE="${SERVICE_CUDA_DEVICE:-${CUDA_DEVICE:-0}}"

export QWEN_BACKEND_BASE_URL="${QWEN_BACKEND_BASE_URL:-http://127.0.0.1:18080}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${SERVICE_CUDA_DEVICE}}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export DINO_DEVICE="${DINO_DEVICE:-cuda:0}"
export VECTOR_DEVICE="${VECTOR_DEVICE:-${DINO_DEVICE}}"
export VECTOR_EMBED_MODEL="${VECTOR_EMBED_MODEL:-/share/shared_weights/dinov3/facebook/dinov3-vith16plus-pretrain-lvd1689m}"
export VECTOR_DIM="${VECTOR_DIM:-1280}"
export VECTOR_ES_INDEX="${VECTOR_ES_INDEX:-intour_vector_store}"
export QWEN_MAX_NEW_TOKENS="${QWEN_MAX_NEW_TOKENS:-4}"
export MAX_IMAGE_DIM="${MAX_IMAGE_DIM:-768}"
export RERANK_TOP_K="${RERANK_TOP_K:-1}"
export RERANK_EXECUTION="${RERANK_EXECUTION:-sequential}"
RUN_MODE="${RUN_MODE:-background}"

if [[ "${RUN_MODE}" == "foreground" ]]; then
  exec "${CONDA_PYTHON}" -m uvicorn qwen3_vl_single_gpu.app:app \
    --app-dir "${ROOT_DIR}/src" \
    --host "${HOST}" \
    --port "${PORT}"
fi

nohup "${CONDA_PYTHON}" -m uvicorn qwen3_vl_single_gpu.app:app \
  --app-dir "${ROOT_DIR}/src" \
  --host "${HOST}" \
  --port "${PORT}" \
  > "${LOG_DIR}/wrapper_service.log" 2>&1 &

echo "wrapper service started on ${HOST}:${PORT}"
echo "log: ${LOG_DIR}/wrapper_service.log"
