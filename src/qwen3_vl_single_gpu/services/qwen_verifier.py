from __future__ import annotations

import re
import time
from dataclasses import dataclass

from qwen3_vl_single_gpu.clients.llama_backend import LlamaBackendClient
from qwen3_vl_single_gpu.config import Settings
from qwen3_vl_single_gpu.prompts import ROAMII_PROMPT_V4_SYSTEM, ROAMII_PROMPT_V4_USER
from qwen3_vl_single_gpu.utils.image_io import image_to_data_url, load_image, resize_for_vlm


@dataclass(frozen=True)
class VerificationDecision:
    is_match: bool
    raw_response: str
    latency_ms: float


class Qwen3VLGGUFVerifier:
    def __init__(self, settings: Settings, backend_client: LlamaBackendClient) -> None:
        self.settings = settings
        self.backend_client = backend_client
        self.system_prompt = ROAMII_PROMPT_V4_SYSTEM
        self.user_prompt = ROAMII_PROMPT_V4_USER

    @property
    def model_version(self) -> str:
        return self.settings.qwen_public_model_version

    def _parse_verification_response(self, response_text: str) -> bool:
        lowered = response_text.strip().lower()
        match = re.search(r"\b(yes|no)\b", lowered)
        if match:
            return match.group(1) == "yes"

        yes_pos = lowered.rfind("yes")
        no_pos = lowered.rfind("no")
        return yes_pos > no_pos

    def verify_pair(self, query_image_source: str, candidate_image_source: str) -> VerificationDecision:
        started = time.perf_counter()
        query_image = resize_for_vlm(load_image(query_image_source, self.settings), self.settings.max_image_dim)
        candidate_image = resize_for_vlm(load_image(candidate_image_source, self.settings), self.settings.max_image_dim)

        payload = {
            "model": self.settings.qwen_backend_model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_to_data_url(query_image)}},
                        {"type": "image_url", "image_url": {"url": image_to_data_url(candidate_image)}},
                        {"type": "text", "text": self.user_prompt},
                    ],
                },
            ],
            "temperature": 0,
            "max_tokens": self.settings.qwen_max_new_tokens,
            "stream": False,
        }
        response = self.backend_client.chat_completions(payload)
        return VerificationDecision(
            is_match=self._parse_verification_response(response.raw_text),
            raw_response=response.raw_text,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )
