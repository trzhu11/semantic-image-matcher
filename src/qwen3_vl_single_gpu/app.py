from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import time

import requests
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from qwen3_vl_single_gpu.api.schemas import AddImageRequest, EncodeRequest, RerankRequest, SearchVectorRequest
from qwen3_vl_single_gpu.clients.llama_backend import LlamaBackendClient
from qwen3_vl_single_gpu.config import Settings, load_settings
from qwen3_vl_single_gpu.models.dino_encoder import DINOEncoder
from qwen3_vl_single_gpu.services.chat_proxy import ChatProxyService
from qwen3_vl_single_gpu.services.qwen_verifier import Qwen3VLGGUFVerifier
from qwen3_vl_single_gpu.services.rerank import RerankService, SequentialVerificationRunner, TopKSelectionStrategy
from qwen3_vl_single_gpu.services.vector_store import ElasticsearchVectorStore
from qwen3_vl_single_gpu.utils.image_io import load_image


LEGACY_ERROR_PATHS = frozenset({"/add_image", "/search_vector", "/stats"})


class ServiceContainer:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.backend_client = LlamaBackendClient(settings)
        self.chat_proxy = ChatProxyService(settings, self.backend_client)
        self.dino_encoder = DINOEncoder(settings)
        self.vector_store = ElasticsearchVectorStore(settings, self.dino_encoder)
        self.qwen_verifier = Qwen3VLGGUFVerifier(settings, self.backend_client)
        if settings.rerank_execution != "sequential":
            raise ValueError(f"unsupported rerank execution mode: {settings.rerank_execution}")
        self.rerank_service = RerankService(
            verifier=self.qwen_verifier,
            selection_strategy=TopKSelectionStrategy(settings.rerank_top_k),
            verification_runner=SequentialVerificationRunner(),
        )

    def load(self) -> None:
        self.backend_client.health()
        self.dino_encoder.load()
        self.vector_store.load()


settings = load_settings()
container = ServiceContainer(settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await asyncio.to_thread(container.load)
    app.state.container = container
    yield


app = FastAPI(title="Qwen3-VL Single GPU Wrapper", version="0.1.0", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    if request.url.path in LEGACY_ERROR_PATHS:
        return JSONResponse(status_code=422, content={"detail": exc.errors()})

    errors = exc.errors()
    message = "invalid request"
    if errors:
        message = str(errors[0].get("msg", message))
        if message.startswith("Value error, "):
            message = message[len("Value error, ") :]
    return JSONResponse(
        status_code=400,
        content={
            "code": 400,
            "message": message,
            "data": None,
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    if request.url.path in LEGACY_ERROR_PATHS:
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
        )

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "code": exc.status_code,
            "message": str(exc.detail),
            "data": None,
        },
    )


@app.get("/health")
async def health_check():
    backend_health = await asyncio.to_thread(container.backend_client.health)
    return {
        "status": "healthy",
        "backend": backend_health,
        "vectorBackend": "elasticsearch",
        "vectorIndex": container.settings.vector_es_index,
        "vectorEsPing": container.vector_store.ping(),
        "vectorNextIndex": container.vector_store.next_vector_index,
        "dinoModelVersion": container.dino_encoder.model_version,
        "dinoDevice": container.settings.dino_device,
        "vectorDim": container.settings.vector_dim,
        "qwenModelVersion": container.qwen_verifier.model_version,
        "rerankTopK": container.settings.rerank_top_k,
        "rerankExecution": container.settings.rerank_execution,
        "benchmarkDatasetDir": container.settings.benchmark_dataset_dir,
    }


@app.get("/v1/models")
async def list_models():
    return await asyncio.to_thread(container.backend_client.list_models)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid json payload: {exc}") from exc

    try:
        return await asyncio.to_thread(container.chat_proxy.forward_chat_payload, payload)
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"backend request failed: {exc}") from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"chat proxy failed: {exc}") from exc


@app.post("/add_image")
async def add_image(request: AddImageRequest):
    try:
        faiss_index = await asyncio.to_thread(container.vector_store.add_image, request.image_base64)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"invalid image input: {exc}") from exc
    except RuntimeError as exc:
        detail = str(exc)
        status_code = 503 if detail == "vector db is not ready" else 500
        raise HTTPException(status_code=status_code, detail=detail) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"failed to persist vector: {exc}") from exc

    return {
        "status": "success",
        "faiss_index": faiss_index,
    }


@app.post("/search_vector")
async def search_vector(request: SearchVectorRequest):
    try:
        indices = await asyncio.to_thread(
            container.vector_store.search,
            request.image_base64,
            request.k,
            request.index_list,
        )
    except ValueError as exc:
        detail = str(exc)
        if detail == "k must be > 0":
            raise HTTPException(status_code=400, detail=detail) from exc
        raise HTTPException(status_code=400, detail=f"invalid image input: {exc}") from exc
    except RuntimeError as exc:
        detail = str(exc)
        status_code = 503 if detail == "vector db is not ready" else 500
        raise HTTPException(status_code=status_code, detail=detail) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"search failed: {exc}") from exc

    return {
        "status": "success",
        "index": indices,
    }


@app.post("/vector/encode")
async def vector_encode(
    request: EncodeRequest,
    include_timings: bool = Query(default=False, alias="includeTimings"),
):
    started = time.perf_counter()
    try:
        image = load_image(request.imageUrl, container.settings)
        vector = await asyncio.to_thread(container.dino_encoder.encode_image, image)
    except requests.RequestException as exc:
        raise HTTPException(status_code=422, detail=f"failed to download image: {exc}") from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"invalid image input: {exc}") from exc

    data = {
        "vector": vector,
        "dim": len(vector),
        "modelVersion": container.dino_encoder.model_version,
    }
    if include_timings:
        data["timings"] = {
            "dinoEncodeMs": round((time.perf_counter() - started) * 1000, 2),
        }

    return {
        "code": 200,
        "message": "success",
        "data": data,
    }


@app.get("/stats")
async def stats():
    try:
        return await asyncio.to_thread(container.vector_store.stats)
    except RuntimeError as exc:
        detail = str(exc)
        status_code = 503 if detail == "vector db is not ready" else 500
        raise HTTPException(status_code=status_code, detail=detail) from exc


@app.post("/vector/rerank")
async def vector_rerank(
    request: RerankRequest,
    include_timings: bool = Query(default=False, alias="includeTimings"),
):
    started = time.perf_counter()
    try:
        result = await asyncio.to_thread(
            container.rerank_service.rerank,
            request.queryImageUrl,
            request.candidateImageUrls,
        )
    except requests.RequestException as exc:
        raise HTTPException(status_code=422, detail=f"failed to download image: {exc}") from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=422, detail=f"invalid image input: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"rerank failed: {exc}") from exc

    data = {
        "matchedImageUrl": result.matched_image_url,
        "matchedIndex": result.matched_index,
        "checkedCount": result.checked_count,
        "isMatch": result.is_match,
        "strategy": result.strategy,
        "modelVersion": result.model_version,
    }
    if include_timings:
        data["timings"] = {
            "vlmMs": result.vlm_latency_ms,
            "totalMs": round((time.perf_counter() - started) * 1000, 2),
        }

    return {
        "code": 200,
        "message": "success",
        "data": data,
    }
