from __future__ import annotations

import base64
import io
from dataclasses import replace
from unittest.mock import patch

from PIL import Image

from qwen3_vl_single_gpu.config import Settings
from qwen3_vl_single_gpu.services.vector_store import ElasticsearchVectorStore


class FakeEncoder:
    def __init__(self) -> None:
        self.calls = 0

    def encode_image(self, image: Image.Image) -> list[float]:
        self.calls += 1
        return [0.1, 0.2, 0.3]


class FakeIndices:
    def __init__(self) -> None:
        self.created: list[tuple[str, dict]] = []
        self.exists_result = True

    def exists(self, *, index: str) -> bool:
        return self.exists_result

    def create(self, *, index: str, mappings: dict) -> None:
        self.created.append((index, mappings))


class FakeElasticsearchClient:
    def __init__(self) -> None:
        self.indices = FakeIndices()
        self.index_calls: list[dict] = []
        self.search_calls: list[dict] = []

    def ping(self) -> bool:
        return True

    def index(self, *, index: str, id: str, document: dict, refresh: str) -> None:
        self.index_calls.append(
            {
                "index": index,
                "id": id,
                "document": document,
                "refresh": refresh,
            }
        )

    def search(self, *, index: str, body: dict) -> dict:
        self.search_calls.append({"index": index, "body": body})
        return {
            "hits": {
                "hits": [
                    {"_source": {"faiss_index": 9}},
                    {"_source": {"faiss_index": 4}},
                ]
            }
        }

    def count(self, *, index: str) -> dict:
        return {"count": 12}


def make_settings() -> Settings:
    return Settings(
        image_timeout_sec=15.0,
        image_url_rewrite_from="",
        image_url_rewrite_to="",
        image_http_trust_env=False,
        image_force_direct_hosts=frozenset(),
        dino_model_path="/tmp/dino",
        dino_device="cpu",
        vector_dim=3,
        es_host="https://example.com:9200",
        es_user="elastic",
        es_pass="secret",
        es_cert="/tmp/http.pem",
        es_verify_certs=True,
        vector_es_index="intour_vector_store",
        vector_similarity="cosine",
        qwen_backend_base_url="http://127.0.0.1:18080",
        qwen_backend_timeout_sec=120.0,
        qwen_backend_model="qwen",
        qwen_public_model_version="Qwen3-VL-8B-Instruct",
        qwen_max_new_tokens=4,
        max_image_dim=768,
        rerank_top_k=1,
        rerank_execution="sequential",
        benchmark_dataset_dir="/tmp/bench",
    )


def make_base64_image() -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), color="red").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def test_add_image_persists_vector_with_incrementing_index() -> None:
    store = ElasticsearchVectorStore(make_settings(), FakeEncoder())
    client = FakeElasticsearchClient()
    store.client = client
    store._next_vector_index = 5

    faiss_index = store.add_image(make_base64_image())

    assert faiss_index == 5
    assert store.next_vector_index == 6
    assert len(client.index_calls) == 1
    call = client.index_calls[0]
    assert call["index"] == "intour_vector_store"
    assert call["id"] == "5"
    assert call["refresh"] == "wait_for"
    assert call["document"]["faiss_index"] == 5
    assert call["document"]["image_vector"] == [0.1, 0.2, 0.3]


def test_search_uses_knn_filter_and_returns_indices() -> None:
    store = ElasticsearchVectorStore(make_settings(), FakeEncoder())
    client = FakeElasticsearchClient()
    store.client = client

    indices = store.search(make_base64_image(), k=3, index_list=[1, 2, 3])

    assert indices == [9, 4]
    assert len(client.search_calls) == 1
    query = client.search_calls[0]["body"]["query"]["knn"]
    assert query["k"] == 3
    assert query["filter"] == {"terms": {"faiss_index": [1, 2, 3]}}
    assert query["query_vector"] == [0.1, 0.2, 0.3]


def test_stats_returns_legacy_shape() -> None:
    store = ElasticsearchVectorStore(make_settings(), FakeEncoder())
    client = FakeElasticsearchClient()
    store.client = client
    store._next_vector_index = 7

    stats = store.stats()

    assert stats == {
        "backend": "elasticsearch",
        "index": "intour_vector_store",
        "count": 12,
        "next_vector_index": 7,
    }


def test_load_omits_auth_and_cert_when_optional_es_settings_are_empty() -> None:
    settings = replace(
        make_settings(),
        es_host="http://127.0.0.1:9200",
        es_user="",
        es_pass="",
        es_cert="",
        es_verify_certs=False,
    )
    captured: dict = {}
    fake_client = FakeElasticsearchClient()

    def build_client(**kwargs):
        captured.update(kwargs)
        return fake_client

    with patch("qwen3_vl_single_gpu.services.vector_store.Elasticsearch", side_effect=build_client):
        store = ElasticsearchVectorStore(settings, FakeEncoder())
        store.load()

    assert captured["hosts"] == ["http://127.0.0.1:9200"]
    assert captured["verify_certs"] is False
    assert captured["request_timeout"] == 60
    assert "basic_auth" not in captured
    assert "ca_certs" not in captured
