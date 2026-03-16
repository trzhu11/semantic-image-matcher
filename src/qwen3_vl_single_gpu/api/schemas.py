from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field, field_validator


class EncodeRequest(BaseModel):
    imageUrl: str = Field(min_length=1)


class AddImageRequest(BaseModel):
    image_base64: str = Field(min_length=1)


class SearchVectorRequest(BaseModel):
    image_base64: str = Field(min_length=1)
    index_list: List[int] | None = None
    k: int = Field(gt=0)


class RerankRequest(BaseModel):
    queryImageUrl: str = Field(min_length=1)
    candidateImageUrls: List[str]

    @field_validator("candidateImageUrls")
    @classmethod
    def validate_candidates(cls, value: List[str]) -> List[str]:
        cleaned = [item.strip() for item in value if isinstance(item, str) and item.strip()]
        if not cleaned:
            raise ValueError("candidateImageUrls must not be empty")
        return cleaned
