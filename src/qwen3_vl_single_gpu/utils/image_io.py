from __future__ import annotations

import base64
import io
from pathlib import Path
from urllib.parse import unquote_to_bytes, urlsplit

import requests
from PIL import Image

from qwen3_vl_single_gpu.config import Settings


def _decode_data_url_to_image(data_url: str) -> Image.Image:
    header, _, payload = data_url.partition(",")
    if not payload:
        raise ValueError("invalid data url")
    if ";base64" in header:
        raw = base64.b64decode(payload)
    else:
        raw = unquote_to_bytes(payload)
    with Image.open(io.BytesIO(raw)) as image:
        return image.convert("RGB")


def rewrite_image_url(url: str, settings: Settings) -> str:
    if settings.image_url_rewrite_from and url.startswith(settings.image_url_rewrite_from):
        return settings.image_url_rewrite_to + url[len(settings.image_url_rewrite_from) :]
    return url


def load_image(source: str, settings: Settings) -> Image.Image:
    if source.startswith("data:image/"):
        return _decode_data_url_to_image(source)

    if source.startswith(("http://", "https://")):
        real_url = rewrite_image_url(source, settings)
        parts = urlsplit(real_url)
        origin = f"{parts.scheme}://{parts.netloc}/" if parts.scheme and parts.netloc else ""
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
            ),
            "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }
        if origin:
            headers["Referer"] = origin

        request_kwargs = {
            "headers": headers,
            "timeout": settings.image_timeout_sec,
        }
        if (parts.hostname or "").lower() in settings.image_force_direct_hosts:
            request_kwargs["proxies"] = {"http": None, "https": None}

        with requests.Session() as session:
            session.trust_env = settings.image_http_trust_env
            response = session.get(real_url, **request_kwargs)
        response.raise_for_status()
        with Image.open(io.BytesIO(response.content)) as image:
            return image.convert("RGB")

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"image not found: {source}")
    with Image.open(path) as image:
        return image.convert("RGB")


def resize_for_vlm(image: Image.Image, max_image_dim: int) -> Image.Image:
    resized = image.copy()
    resized.thumbnail((max_image_dim, max_image_dim), Image.Resampling.LANCZOS)
    return resized


def image_to_data_url(image: Image.Image, *, format_name: str = "JPEG", quality: int = 90) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format_name, quality=quality)
    mime = "image/jpeg" if format_name.upper() == "JPEG" else f"image/{format_name.lower()}"
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:{mime};base64,{encoded}"
