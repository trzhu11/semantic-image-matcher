from __future__ import annotations

import base64
import binascii
import io
import threading
from datetime import datetime, timezone
from typing import Any

from elasticsearch import Elasticsearch
from PIL import Image

from qwen3_vl_single_gpu.config import Settings
from qwen3_vl_single_gpu.models.dino_encoder import DINOEncoder


class ElasticsearchVectorStore:
    def __init__(self, settings: Settings, encoder: DINOEncoder) -> None:
        self.settings = settings
        self.encoder = encoder
        self.client: Elasticsearch | None = None
        self._setup_lock = threading.Lock()
        self._index_lock = threading.Lock()
        self._next_vector_index = 0

    @property
    def next_vector_index(self) -> int:
        return self._next_vector_index

    @property
    def index_name(self) -> str:
        return self.settings.vector_es_index

    def load(self) -> None:
        if self.client is not None:
            return

        with self._setup_lock:
            if self.client is not None:
                return

            client_kwargs: dict[str, Any] = {
                "hosts": [self.settings.es_host],
                "verify_certs": self.settings.es_verify_certs,
                "request_timeout": 60,
            }
            if self.settings.es_user or self.settings.es_pass:
                client_kwargs["basic_auth"] = (self.settings.es_user, self.settings.es_pass)
            if self.settings.es_cert:
                client_kwargs["ca_certs"] = self.settings.es_cert

            client = Elasticsearch(**client_kwargs)
            if not client.ping():
                raise RuntimeError(f"cannot connect Elasticsearch: {self.settings.es_host}")

            self._ensure_index(client)
            self._next_vector_index = self._load_next_index(client)
            self.client = client

    def ping(self) -> bool:
        if self.client is None:
            return False
        return bool(self.client.ping())

    def add_image(self, image_base64: str) -> int:
        client = self._require_client()
        image = self._decode_base64_to_image(image_base64)
        vector = self.encoder.encode_image(image)

        with self._index_lock:
            faiss_index = self._next_vector_index
            self._next_vector_index += 1

        doc = {
            "faiss_index": faiss_index,
            "image_vector": vector,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        client.index(index=self.index_name, id=str(faiss_index), document=doc, refresh="wait_for")
        return faiss_index

    def search(self, image_base64: str, k: int, index_list: list[int] | None = None) -> list[int]:
        if k <= 0:
            raise ValueError("k must be > 0")

        client = self._require_client()
        image = self._decode_base64_to_image(image_base64)
        vector = self.encoder.encode_image(image)

        terms_filter = None
        if index_list:
            terms_filter = {"terms": {"faiss_index": index_list}}

        knn_query: dict[str, Any] = {
            "field": "image_vector",
            "query_vector": vector,
            "k": k,
            "num_candidates": max(k * 20, 100),
        }
        if terms_filter:
            knn_query["filter"] = terms_filter

        result = client.search(
            index=self.index_name,
            body={
                "size": k,
                "_source": ["faiss_index"],
                "query": {"knn": knn_query},
            },
        )
        hits = result.get("hits", {}).get("hits", [])
        return [
            int(hit.get("_source", {}).get("faiss_index"))
            for hit in hits
            if "faiss_index" in hit.get("_source", {})
        ]

    def stats(self) -> dict[str, Any]:
        client = self._require_client()
        count = client.count(index=self.index_name).get("count", 0)
        return {
            "backend": "elasticsearch",
            "index": self.index_name,
            "count": count,
            "next_vector_index": self.next_vector_index,
        }

    def _require_client(self) -> Elasticsearch:
        self.load()
        if self.client is None:
            raise RuntimeError("vector db is not ready")
        return self.client

    def _ensure_index(self, client: Elasticsearch) -> None:
        mapping = {
            "properties": {
                "faiss_index": {"type": "integer"},
                "image_vector": {
                    "type": "dense_vector",
                    "dims": self.settings.vector_dim,
                    "index": True,
                    "similarity": self.settings.vector_similarity,
                    "index_options": {"type": "int8_hnsw", "m": 16, "ef_construction": 100},
                },
                "created_at": {"type": "date"},
            }
        }
        if not client.indices.exists(index=self.index_name):
            client.indices.create(index=self.index_name, mappings=mapping)

    def _load_next_index(self, client: Elasticsearch) -> int:
        result = client.search(
            index=self.index_name,
            body={
                "size": 1,
                "_source": ["faiss_index"],
                "sort": [{"faiss_index": {"order": "desc"}}],
                "query": {"match_all": {}},
            },
        )
        hits = result.get("hits", {}).get("hits", [])
        if not hits:
            return 0
        max_index = hits[0].get("_source", {}).get("faiss_index", -1)
        return int(max_index) + 1

    def _decode_base64_to_image(self, payload: str) -> Image.Image:
        try:
            raw = base64.b64decode(payload)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(f"invalid base64 payload: {exc}") from exc

        try:
            with Image.open(io.BytesIO(raw)) as image:
                return image.convert("RGB")
        except OSError as exc:
            raise ValueError(f"invalid image payload: {exc}") from exc
