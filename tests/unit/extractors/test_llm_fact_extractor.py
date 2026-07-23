import json

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
    assert "valid JSON object" in messages[0].content
    assert messages[1].role == MessageRole.USER
    assert messages[1].content == "My name is Frank."


def test_extract_calls_client_and_returns_parsed_facts() -> None:
    client = FakeClient(
        response='{"user_name": "Frank"}',
    )

    extractor = LLMFactExtractor(
        client=client,
    )

    result = extractor.extract(
        "My name is Frank.",
    )

    assert result == {
        "user_name": "Frank",
    }

    assert client.call_count == 1
    assert len(client.received_messages) == 2
    assert client.received_messages[0].role == MessageRole.SYSTEM
    assert client.received_messages[1].role == MessageRole.USER


def test_parse_response_returns_facts() -> None:
    extractor = LLMFactExtractor(
        client=FakeClient(),
    )

    result = extractor._parse_response(
        '{"user_name": "Frank", "location": "Hai Phong"}',
    )

    assert result == {
        "user_name": "Frank",
        "location": "Hai Phong",
    }


def test_parse_response_returns_empty_dict() -> None:
    extractor = LLMFactExtractor(
        client=FakeClient(),
    )

    result = extractor._parse_response(
        "{}",
    )

    assert result == {}


def test_parse_response_raises_for_invalid_json() -> None:
    extractor = LLMFactExtractor(
        client=FakeClient(),
    )

    with pytest.raises(json.JSONDecodeError):
        extractor._parse_response(
            "not json",
        )


def test_parse_response_raises_for_json_array() -> None:
    extractor = LLMFactExtractor(
        client=FakeClient(),
    )

    with pytest.raises(
        ValueError,
        match="must be a JSON object",
    ):
        extractor._parse_response(
            '["Frank"]',
        )


def test_parse_response_raises_for_non_string_value() -> None:
    extractor = LLMFactExtractor(
        client=FakeClient(),
    )

    with pytest.raises(
        ValueError,
        match="Fact values must be strings",
    ):
        extractor._parse_response(
            '{"age": 32}',
        )
