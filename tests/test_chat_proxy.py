from qwen3_vl_single_gpu.config import Settings
from qwen3_vl_single_gpu.services.chat_proxy import ChatProxyService


class FakeBackendClient:
    def __init__(self) -> None:
        self.payload = None

    def chat_completions(self, payload):
        self.payload = payload

        class Response:
            def __init__(self, payload):
                self.payload = payload

        return Response(payload)


def _settings() -> Settings:
    return Settings(
        image_timeout_sec=15,
        image_url_rewrite_from="",
        image_url_rewrite_to="",
        image_http_trust_env=False,
        image_force_direct_hosts=frozenset(),
        dino_model_path="/tmp/dino",
        dino_device="cpu",
        qwen_backend_base_url="http://127.0.0.1:18080",
        qwen_backend_timeout_sec=120,
        qwen_backend_model="qwen3-vl-8b-instruct-q8_0",
        qwen_public_model_version="Qwen3-VL-8B-Instruct",
        qwen_max_new_tokens=8,
        max_image_dim=768,
        rerank_top_k=1,
        rerank_execution="sequential",
        benchmark_dataset_dir="/tmp/benchmark",
    )


def test_proxy_adds_default_model_for_text_only_payload() -> None:
    backend = FakeBackendClient()
    service = ChatProxyService(_settings(), backend)

    result = service.forward_chat_payload(
        {
            "messages": [
                {"role": "user", "content": "hello"},
            ],
            "temperature": 0.2,
        }
    )

    assert result["model"] == "qwen3-vl-8b-instruct-q8_0"
    assert backend.payload["model"] == "qwen3-vl-8b-instruct-q8_0"
