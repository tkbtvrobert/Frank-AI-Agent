import pytest

from app.extractors.llm_fact_extractor import LLMFactExtractor
from app.models.message_role import MessageRole
from tests.fakes.fake_client import FakeClient


def test_build_messages_contains_system_and_user_messages() -> None:
    client = FakeClient()
    extractor = LLMFactExtractor(
        client=client,
    )

    messages = extractor._build_messages(
        "My name is Frank.",
    )

    assert len(messages) == 2
    assert messages[0].role == MessageRole.SYSTEM
    assert messages[1].role == MessageRole.USER
    assert messages[1].content == "My name is Frank."


def test_extract_calls_client_with_built_messages() -> None:
    client = FakeClient(
        response='{"user_name": "Frank"}',
    )

    extractor = LLMFactExtractor(
        client=client,
    )

    with pytest.raises(
        NotImplementedError,
        match="Response parsing is not implemented yet",
    ):
        extractor.extract(
            "My name is Frank.",
        )

    assert client.call_count == 1
    assert len(client.received_messages) == 2

    assert (
        client.received_messages[0].role
        == MessageRole.SYSTEM
    )

    assert (
        client.received_messages[1].role
        == MessageRole.USER
    )

    assert (
        client.received_messages[1].content
        == "My name is Frank."
    )


def test_extract_is_not_implemented() -> None:
    client = FakeClient()
    extractor = LLMFactExtractor(
        client=client,
    )

    with pytest.raises(NotImplementedError):
        extractor.extract(
            "My name is Frank.",
        )