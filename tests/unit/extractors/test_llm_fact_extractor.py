import pytest

from app.extractors.llm_fact_extractor import LLMFactExtractor
from tests.fakes.fake_client import FakeClient


def test_extract_is_not_implemented() -> None:
    client = FakeClient()
    extractor = LLMFactExtractor(
        client=client,
    )

    with pytest.raises(NotImplementedError):
        extractor.extract(
            "My name is Frank.",
        )