#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_DIR="${ROOT_DIR}/vendor"
LLAMA_DIR="${LLAMA_CPP_DIR:-${VENDOR_DIR}/llama.cpp}"
CMAKE_BIN="${CMAKE_BIN:-${VENDOR_DIR}/cmake/lib/python3.10/site-packages/cmake/data/bin/cmake}"
CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES:-89}"

mkdir -p "${VENDOR_DIR}"

if [[ ! -d "${LLAMA_DIR}/.git" ]]; then
  git clone https://github.com/ggml-org/llama.cpp.git "${LLAMA_DIR}"
fi

cd "${LLAMA_DIR}"

if [[ ! -x "${CMAKE_BIN}" ]]; then
  CMAKE_BIN="$(command -v cmake || true)"
fi

if [[ -z "${CMAKE_BIN}" ]]; then
  echo "cmake not found. Install it into ${VENDOR_DIR}/cmake first." >&2
  exit 1
fi

"${CMAKE_BIN}" -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" -DCMAKE_BUILD_TYPE=Release
"${CMAKE_BIN}" --build build --config Release --target llama-server -j"$(nproc)"

if [[ -x "${LLAMA_DIR}/build/bin/llama-server" ]]; then
  echo "llama.cpp build finished: ${LLAMA_DIR}/build/bin/llama-server"
elif [[ -x "${LLAMA_DIR}/llama-server" ]]; then
  echo "llama.cpp build finished: ${LLAMA_DIR}/llama-server"
else
  echo "llama.cpp build finished, but llama-server binary path needs manual confirmation." >&2
fi
