from __future__ import annotations

import os
import threading
from typing import List

import torch
from PIL import Image

from qwen3_vl_single_gpu.config import Settings


class DINOEncoder:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.processor = None
        self.model = None
        self.model_dtype = torch.float32
        self._lock = threading.Lock()

    @property
    def model_version(self) -> str:
        return os.path.basename(self.settings.dino_model_path.rstrip("/"))

    def load(self) -> None:
        if self.processor is not None and self.model is not None:
            return

        with self._lock:
            if self.processor is not None and self.model is not None:
                return

            from transformers import AutoImageProcessor, AutoModel

            self.processor = AutoImageProcessor.from_pretrained(self.settings.dino_model_path)
            load_dtype = torch.float32
            if self.settings.dino_device.startswith("cuda"):
                load_dtype = torch.bfloat16
            self.model = AutoModel.from_pretrained(
                self.settings.dino_model_path,
                dtype=load_dtype,
            ).to(self.settings.dino_device)
            self.model.eval()
            self.model_dtype = next(self.model.parameters()).dtype

    def encode_image(self, image: Image.Image) -> List[float]:
        self.load()
        if self.processor is None or self.model is None:
            raise RuntimeError("DINO encoder is not initialized")

        with torch.inference_mode():
            inputs = self.processor(images=image, return_tensors="pt").to(self.settings.dino_device)
            if self.settings.dino_device.startswith("cuda") and "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(self.model_dtype)
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].detach().float().cpu().numpy()[0]
        if embedding.shape[0] != self.settings.vector_dim:
            raise RuntimeError(
                f"vector dim mismatch, expected={self.settings.vector_dim}, got={embedding.shape[0]}"
            )
        return embedding.tolist()
