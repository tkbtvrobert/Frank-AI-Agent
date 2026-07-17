from unittest.mock import MagicMock, call, patch
import httpx
from groq import APIConnectionError
from app.clients.groq_client import GroqClient
from app.config_models.retry_config import RetryConfig
import pytest
from app.exceptions.client_exceptions import ClientConnectionError
from groq import AuthenticationError
from app.exceptions.client_exceptions import ClientAuthenticationError


@patch("app.clients.groq_client.time.sleep")
def test_chat_retries_connection_error_then_succeeds(mock_sleep: MagicMock) -> None:
    retry_config = RetryConfig(
        max_attempts=3,
        initial_delay_seconds=1.0,
        backoff_multiplier=2.0,
    )

    client = GroqClient(
        retry_config=retry_config,
    )

    request = httpx.Request(
        method="POST",
        url="https://api.groq.com/openai/v1/chat/completions",
    )

    connection_error = APIConnectionError(
        request=request,
    )

    successful_response = MagicMock()
    successful_response.choices[0].message.content = "Hello, Frank!"

    mock_create = MagicMock(
        side_effect=[
            connection_error,
            connection_error,
            successful_response,
        ]
    )

    client.client.chat.completions.create = mock_create

    result = client.chat(
        messages=[
            {
                "role": "user",
                "content": "Hello",
            }
        ]
    )

    assert result == "Hello, Frank!"
    assert mock_create.call_count == 3

    assert mock_sleep.call_count == 2
    mock_sleep.assert_any_call(1.0)
    mock_sleep.assert_any_call(2.0)

    assert mock_sleep.call_args_list == [
        call(1.0),
        call(2.0),
    ]


def test_chat_raises_connection_error_after_max_attempts() -> None:
    retry_config = RetryConfig(
        max_attempts=3,
        initial_delay_seconds=0,
    )

    client = GroqClient(
        retry_config=retry_config,
    )

    request = httpx.Request(
        method="POST",
        url="https://api.groq.com/openai/v1/chat/completions",
    )

    connection_error = APIConnectionError(
        request=request,
    )

    mock_create = MagicMock(
        side_effect=connection_error,
    )

    client.client.chat.completions.create = mock_create

    with pytest.raises(ClientConnectionError):
        client.chat(
            messages=[
                {
                    "role": "user",
                    "content": "Hello",
                }
            ]
        )

    assert mock_create.call_count == 3


def test_chat_does_not_retry_authentication_error() -> None:
    retry_config = RetryConfig(
        max_attempts=3,
        initial_delay_seconds=0,
    )

    client = GroqClient(
        retry_config=retry_config,
    )

    request = httpx.Request(
        method="POST",
        url="https://api.groq.com/openai/v1/chat/completions",
    )

    response = httpx.Response(
        status_code=401,
        request=request,
    )

    authentication_error = AuthenticationError(
        "Invalid API Key",
        response=response,
        body={
            "error": {
                "message": "Invalid API Key",
            }
        },
    )

    mock_create = MagicMock(
        side_effect=authentication_error,
    )

    client.client.chat.completions.create = mock_create

    with pytest.raises(ClientAuthenticationError):
        client.chat(
            messages=[
                {
                    "role": "user",
                    "content": "Hello",
                }
            ]
        )

    assert mock_create.call_count == 1


def test_calculate_delay_uses_exponential_backoff() -> None:
    retry_config = RetryConfig(
        max_attempts=4,
        initial_delay_seconds=1.0,
        backoff_multiplier=2.0,
    )

    client = GroqClient(
        retry_config=retry_config,
    )

    assert client._calculate_delay(1) == 1.0
    assert client._calculate_delay(2) == 2.0
    assert client._calculate_delay(3) == 4.0
