from __future__ import annotations

from copy import deepcopy
from typing import Any

from qwen3_vl_single_gpu.clients.llama_backend import LlamaBackendClient
from qwen3_vl_single_gpu.config import Settings
from qwen3_vl_single_gpu.utils.image_io import image_to_data_url, load_image, resize_for_vlm


class ChatProxyService:
    def __init__(self, settings: Settings, backend_client: LlamaBackendClient) -> None:
        self.settings = settings
        self.backend_client = backend_client

    def forward_chat_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        normalized = self._normalize_payload(payload)
        return self.backend_client.chat_completions(normalized).payload

    def _normalize_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        normalized = deepcopy(payload)
        normalized.setdefault("model", self.settings.qwen_backend_model)
        messages = normalized.get("messages")
        if not isinstance(messages, list):
            return normalized

        for message in messages:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if not isinstance(content, list):
                continue
            message["content"] = [self._normalize_content_item(item) for item in content]
        return normalized

    def _normalize_content_item(self, item: Any) -> Any:
        if not isinstance(item, dict):
            return item

        item_type = item.get("type")
        if item_type not in {"image_url", "input_image"}:
            return item

        image_url = item.get("image_url")
        if isinstance(image_url, dict):
            url = image_url.get("url", "")
        else:
            url = image_url or item.get("url", "")

        if not isinstance(url, str) or not url:
            return item
        if url.startswith("data:image/"):
            return item

        image = resize_for_vlm(load_image(url, self.settings), self.settings.max_image_dim)
        data_url = image_to_data_url(image)
        if item_type == "input_image":
            return {"type": "input_image", "image_url": {"url": data_url}}
        return {"type": "image_url", "image_url": {"url": data_url}}
