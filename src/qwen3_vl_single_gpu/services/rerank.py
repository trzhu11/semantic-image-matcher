from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from qwen3_vl_single_gpu.services.qwen_verifier import VerificationDecision


@dataclass(frozen=True)
class CandidateItem:
    index: int
    image_url: str


@dataclass(frozen=True)
class VerificationAttempt:
    candidate: CandidateItem
    is_match: bool
    raw_response: str
    latency_ms: float


@dataclass(frozen=True)
class RerankOutcome:
    matched_image_url: str | None
    matched_index: int | None
    checked_count: int
    is_match: bool
    strategy: str
    model_version: str
    vlm_latency_ms: float


class PairVerifier(Protocol):
    @property
    def model_version(self) -> str:
        ...

    def verify_pair(self, query_image_source: str, candidate_image_source: str) -> VerificationDecision:
        ...


class CandidateSelectionStrategy(Protocol):
    @property
    def name(self) -> str:
        ...

    def select(self, candidate_urls: Sequence[str]) -> list[CandidateItem]:
        ...


class VerificationRunner(Protocol):
    @property
    def name(self) -> str:
        ...

    def run(
        self,
        query_image_source: str,
        candidates: Sequence[CandidateItem],
        verifier: PairVerifier,
    ) -> list[VerificationAttempt]:
        ...


class TopKSelectionStrategy:
    def __init__(self, top_k: int) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be > 0")
        self.top_k = top_k

    @property
    def name(self) -> str:
        return f"top{self.top_k}_ordered"

    def select(self, candidate_urls: Sequence[str]) -> list[CandidateItem]:
        return [
            CandidateItem(index=index, image_url=url)
            for index, url in enumerate(candidate_urls[: self.top_k])
        ]


class SequentialVerificationRunner:
    @property
    def name(self) -> str:
        return "sequential"

    def run(
        self,
        query_image_source: str,
        candidates: Sequence[CandidateItem],
        verifier: PairVerifier,
    ) -> list[VerificationAttempt]:
        attempts: list[VerificationAttempt] = []
        for candidate in candidates:
            decision = verifier.verify_pair(query_image_source, candidate.image_url)
            attempt = VerificationAttempt(
                candidate=candidate,
                is_match=decision.is_match,
                raw_response=decision.raw_response,
                latency_ms=decision.latency_ms,
            )
            attempts.append(attempt)
            if attempt.is_match:
                break
        return attempts


class RerankService:
    def __init__(
        self,
        verifier: PairVerifier,
        selection_strategy: CandidateSelectionStrategy,
        verification_runner: VerificationRunner,
    ) -> None:
        self.verifier = verifier
        self.selection_strategy = selection_strategy
        self.verification_runner = verification_runner

    def rerank(self, query_image_source: str, candidate_urls: Sequence[str]) -> RerankOutcome:
        selected_candidates = self.selection_strategy.select(candidate_urls)
        attempts = self.verification_runner.run(
            query_image_source=query_image_source,
            candidates=selected_candidates,
            verifier=self.verifier,
        )

        matched_attempt = next((attempt for attempt in attempts if attempt.is_match), None)
        if matched_attempt is None:
            return RerankOutcome(
                matched_image_url=None,
                matched_index=None,
                checked_count=len(attempts),
                is_match=False,
                strategy=self.selection_strategy.name,
                model_version=self.verifier.model_version,
                vlm_latency_ms=round(sum(attempt.latency_ms for attempt in attempts), 2),
            )

        return RerankOutcome(
            matched_image_url=matched_attempt.candidate.image_url,
            matched_index=matched_attempt.candidate.index,
            checked_count=len(attempts),
            is_match=True,
            strategy=self.selection_strategy.name,
            model_version=self.verifier.model_version,
            vlm_latency_ms=round(sum(attempt.latency_ms for attempt in attempts), 2),
        )
