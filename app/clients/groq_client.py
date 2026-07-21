"""
GroqClient is responsible for:

1. Communicating with the Groq API

2. Sending messages

3. Returning LLM responses

4. Handling API-level exceptions

Not responsible for:

- Managing History

- Managing Prompts

- Managing Workflows
"""

import logging
import time

from groq import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    Groq,
    RateLimitError,
)

from app.clients.base_client import BaseClient
from app.config_models.groq_config import GroqConfig
from app.config_models.retry_config import RetryConfig
from app.exceptions.client_exceptions import (
    ClientAuthenticationError,
    ClientConnectionError,
    ClientRateLimitError,
    ClientTimeoutError,
)
from app.models.message import Message

logger = logging.getLogger(__name__)


class GroqClient(BaseClient):
    def __init__(self, groq_config: GroqConfig, retry_config: RetryConfig) -> None:
        self.client = Groq(api_key=groq_config.api_key)
        self.groq_config = groq_config
        self.retry_config = retry_config

    def _calculate_delay(self, attempt: int) -> float:
        return (
            self.retry_config.initial_delay_seconds
            * self.retry_config.backoff_multiplier ** (attempt - 1)
        )

    def _handle_retryable_error(
        self,
        *,
        attempt: int,
        max_attempts: int,
        error: Exception,
        error_name: str,
        final_exception: Exception,
    ) -> None:
        if attempt == max_attempts:
            logger.exception(
                "Groq %s failed after %d attempts",
                error_name,
                max_attempts,
            )

            raise final_exception from error

        delay_seconds = self._calculate_delay(attempt)

        logger.warning(
            "Groq %s failed on attempt %d/%d. Retrying in %.1f seconds. Reason: %s",
            error_name,
            attempt,
            max_attempts,
            delay_seconds,
            error,
        )

        time.sleep(delay_seconds)

    def _is_valid_response(self, content: str | None) -> bool:
        if content is None:
            return False

        cleaned_content = content.strip()

        if not cleaned_content:
            return False

        return any(character.isalnum() for character in cleaned_content)
    
    def _format_messages(
        self,
        messages: list[Message],
    ) -> list[dict[str, str]]:
        return [
            {
                "role": message.role.value,
                "content": message.content,
            }
            for message in messages
        ]

    def chat(self, messages: list[Message]) -> str:
        formatted_messages = self._format_messages(messages)

        max_attempts = self.retry_config.max_attempts

        for attempt in range(1, max_attempts + 1):
            try:
                logger.debug(
                    "Sending Groq request attempt=%d/%d",
                    attempt,
                    max_attempts,
                )

                response = self.client.chat.completions.create(
                    messages=formatted_messages,
                    model=self.groq_config.model,
                    temperature=0,
                )

                content = response.choices[0].message.content

                if not self._is_valid_response(content):
                    logger.warning(
                        "Groq returned invalid content on attempt %d/%d: %r",
                        attempt,
                        max_attempts,
                        content,
                    )

                    if attempt == max_attempts:
                        raise RuntimeError(
                            "Groq returned invalid content "
                            f"after {max_attempts} attempts"
                        )

                    delay_seconds = self._calculate_delay(attempt)
                    time.sleep(delay_seconds)
                    continue

                logger.info(
                    "Groq request succeeded on attempt %d/%d",
                    attempt,
                    max_attempts,
                )

                return content
            except AuthenticationError as error:
                logger.exception("Groq authentication failed")

                raise ClientAuthenticationError(
                    "AI client authentication failed"
                ) from error

            except RateLimitError as error:
                self._handle_retryable_error(
                    attempt=attempt,
                    max_attempts=max_attempts,
                    error=error,
                    error_name="rate limit",
                    final_exception=ClientRateLimitError(
                        "AI service rate limit exceeded"
                    ),
                )

            except APITimeoutError as error:
                self._handle_retryable_error(
                    attempt=attempt,
                    max_attempts=max_attempts,
                    error=error,
                    error_name="request timeout",
                    final_exception=ClientTimeoutError("AI client request timed out"),
                )

            except APIConnectionError as error:
                self._handle_retryable_error(
                    attempt=attempt,
                    max_attempts=max_attempts,
                    error=error,
                    error_name="connection",
                    final_exception=ClientConnectionError(
                        "Failed to connect to AI service"
                    ),
                )

        raise RuntimeError("Retry loop ended unexpectedly")
