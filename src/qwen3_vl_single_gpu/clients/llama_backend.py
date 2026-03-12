from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests

from qwen3_vl_single_gpu.config import Settings


@dataclass(frozen=True)
class BackendResponse:
    payload: dict[str, Any]
    raw_text: str


class LlamaBackendClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def _session(self) -> requests.Session:
        session = requests.Session()
        session.trust_env = False
        return session

    @property
    def base_url(self) -> str:
        return self.settings.qwen_backend_base_url.rstrip("/")

    def health(self) -> dict[str, Any]:
        health_url = f"{self.base_url}/health"
        try:
            with self._session() as session:
                response = session.get(health_url, timeout=5)
                response.raise_for_status()
                return response.json()
        except Exception:
            return self.list_models()

    def list_models(self) -> dict[str, Any]:
        with self._session() as session:
            response = session.get(f"{self.base_url}/v1/models", timeout=5)
            response.raise_for_status()
            return response.json()

    def chat_completions(self, payload: dict[str, Any]) -> BackendResponse:
        with self._session() as session:
            response = session.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=self.settings.qwen_backend_timeout_sec,
            )
            response.raise_for_status()
            data = response.json()

        content = ""
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message", {})
            if isinstance(message, dict):
                content = str(message.get("content", "")).strip()

        return BackendResponse(payload=data, raw_text=content)
