from __future__ import annotations

import os
from unittest.mock import patch

from qwen3_vl_single_gpu.config import load_settings


def test_defaults_point_to_dinov3_large_and_es_vector_store() -> None:
    with patch.dict(os.environ, {}, clear=True):
        settings = load_settings()

    assert settings.dino_model_path == "/share/shared_weights/dinov3/facebook/dinov3-vith16plus-pretrain-lvd1689m"
    assert settings.vector_dim == 1280
    assert settings.vector_es_index == "intour_vector_store"
    assert settings.vector_similarity == "cosine"
    assert settings.image_url_rewrite_from == ""
    assert settings.image_url_rewrite_to == ""
    assert settings.es_host == "http://127.0.0.1:9200"
    assert settings.es_user == ""
    assert settings.es_pass == ""
    assert settings.es_cert == ""
    assert settings.es_verify_certs is False


def test_legacy_vector_env_names_are_accepted() -> None:
    with patch.dict(
        os.environ,
        {
            "VECTOR_EMBED_MODEL": "/tmp/dino-legacy",
            "VECTOR_DEVICE": "cpu",
            "VECTOR_DIM": "256",
            "VECTOR_ES_INDEX": "legacy_index",
        },
        clear=True,
    ):
        settings = load_settings()

    assert settings.dino_model_path == "/tmp/dino-legacy"
    assert settings.dino_device == "cpu"
    assert settings.vector_dim == 256
    assert settings.vector_es_index == "legacy_index"
