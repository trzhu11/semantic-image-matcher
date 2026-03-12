from dataclasses import dataclass

from qwen3_vl_single_gpu.services.rerank import RerankService, SequentialVerificationRunner, TopKSelectionStrategy
from qwen3_vl_single_gpu.services.qwen_verifier import VerificationDecision


@dataclass
class FakeVerifier:
    answers: dict[str, bool]
    model_version: str = "fake-qwen"

    def verify_pair(self, query_image_source: str, candidate_image_source: str) -> VerificationDecision:
        is_match = self.answers.get(candidate_image_source, False)
        return VerificationDecision(
            is_match=is_match,
            raw_response="Yes" if is_match else "No",
            latency_ms=12.5,
        )


def test_rerank_returns_first_positive_candidate() -> None:
    service = RerankService(
        verifier=FakeVerifier({"c1": False, "c2": True, "c3": True}),
        selection_strategy=TopKSelectionStrategy(3),
        verification_runner=SequentialVerificationRunner(),
    )

    result = service.rerank("q", ["c1", "c2", "c3"])

    assert result.is_match is True
    assert result.matched_image_url == "c2"
    assert result.matched_index == 1
    assert result.checked_count == 2
    assert result.vlm_latency_ms == 25.0


def test_rerank_returns_no_match_when_all_negative() -> None:
    service = RerankService(
        verifier=FakeVerifier({}),
        selection_strategy=TopKSelectionStrategy(2),
        verification_runner=SequentialVerificationRunner(),
    )

    result = service.rerank("q", ["c1", "c2", "c3"])

    assert result.is_match is False
    assert result.matched_image_url is None
    assert result.matched_index is None
    assert result.checked_count == 2
    assert result.vlm_latency_ms == 25.0
