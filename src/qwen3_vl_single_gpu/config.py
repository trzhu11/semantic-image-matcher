from __future__ import annotations

import os
from dataclasses import dataclass

import torch


def _default_dino_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def _split_hosts(value: str) -> frozenset[str]:
    return frozenset(host.strip().lower() for host in value.split(",") if host.strip())


def _first_env(*names: str, default: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


@dataclass(frozen=True)
class Settings:
    image_timeout_sec: float
    image_url_rewrite_from: str
    image_url_rewrite_to: str
    image_http_trust_env: bool
    image_force_direct_hosts: frozenset[str]
    dino_model_path: str
    dino_device: str
    vector_dim: int
    es_host: str
    es_user: str
    es_pass: str
    es_cert: str
    es_verify_certs: bool
    vector_es_index: str
    vector_similarity: str
    qwen_backend_base_url: str
    qwen_backend_timeout_sec: float
    qwen_backend_model: str
    qwen_public_model_version: str
    qwen_max_new_tokens: int
    max_image_dim: int
    rerank_top_k: int
    rerank_execution: str
    benchmark_dataset_dir: str


def load_settings() -> Settings:
    return Settings(
        image_timeout_sec=float(os.getenv("IMAGE_TIMEOUT_SEC", "15")),
        image_url_rewrite_from=os.getenv("IMAGE_URL_REWRITE_FROM", ""),
        image_url_rewrite_to=os.getenv("IMAGE_URL_REWRITE_TO", ""),
        image_http_trust_env=os.getenv("IMAGE_HTTP_TRUST_ENV", "0") == "1",
        image_force_direct_hosts=_split_hosts(
            os.getenv("IMAGE_FORCE_DIRECT_HOSTS", "127.0.0.1,localhost")
        ),
        dino_model_path=_first_env(
            "DINO_MODEL_PATH",
            "VECTOR_EMBED_MODEL",
            default="/share/shared_weights/dinov3/facebook/dinov3-vith16plus-pretrain-lvd1689m",
        ),
        dino_device=_first_env("DINO_DEVICE", "VECTOR_DEVICE", default=_default_dino_device()),
        vector_dim=int(os.getenv("VECTOR_DIM", "1280")),
        es_host=os.getenv("ES_HOST", "http://127.0.0.1:9200"),
        es_user=os.getenv("ES_USER", ""),
        es_pass=os.getenv("ES_PASS", ""),
        es_cert=os.getenv("ES_CERT", ""),
        es_verify_certs=os.getenv("ES_VERIFY_CERTS", "0") == "1",
        vector_es_index=os.getenv("VECTOR_ES_INDEX", "intour_vector_store"),
        vector_similarity=os.getenv("VECTOR_SIMILARITY", "cosine"),
        qwen_backend_base_url=os.getenv("QWEN_BACKEND_BASE_URL", "http://127.0.0.1:18080"),
        qwen_backend_timeout_sec=float(os.getenv("QWEN_BACKEND_TIMEOUT_SEC", "120")),
        qwen_backend_model=os.getenv("QWEN_BACKEND_MODEL", "qwen3-vl-8b-instruct-q8_0"),
        qwen_public_model_version=os.getenv("QWEN_PUBLIC_MODEL_VERSION", "Qwen3-VL-8B-Instruct"),
        qwen_max_new_tokens=max(1, int(os.getenv("QWEN_MAX_NEW_TOKENS", "4"))),
        max_image_dim=max(256, int(os.getenv("MAX_IMAGE_DIM", "768"))),
        rerank_top_k=max(1, int(os.getenv("RERANK_TOP_K", "1"))),
        rerank_execution=os.getenv("RERANK_EXECUTION", "sequential"),
        benchmark_dataset_dir=os.getenv(
            "BENCHMARK_DATASET_DIR",
            "/share/shared_datasets/roamii-benchmark-frozen/merged_all",
        ),
    )
