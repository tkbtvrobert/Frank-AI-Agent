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

    def chat(self, messages: list[dict[str, str]]) -> str:
        max_attempts = self.retry_config.max_attempts
        for attempt in range(1, max_attempts + 1):
            try:
                logger.debug(
                    "Sending Groq request attempt=%d/%d",
                    attempt,
                    max_attempts,
                )

                response = self.client.chat.completions.create(
                    messages=messages, model=self.groq_config.model,
                )

                logger.info(
                    "Groq request succeeded on attempt %d/%d",
                    attempt,
                    max_attempts,
                )

                return response.choices[0].message.content
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
