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

from groq import Groq
from app.config import GROQ_API_KEY
from app.config import GROQ_MODEL
from groq import APITimeoutError, AuthenticationError, APIConnectionError
from app.clients.base_client import BaseClient
import logging
from app.exceptions.client_exceptions import (
    ClientAuthenticationError,
    ClientConnectionError,
    ClientTimeoutError,
)
from app.config_models.retry_config import RetryConfig
import time

logger = logging.getLogger(__name__)


class GroqClient(BaseClient):
    def __init__(self, retry_config: RetryConfig) -> None:
        self.client = Groq(api_key=GROQ_API_KEY)
        self.retry_config = retry_config

    def _calculate_delay(self, attempt: int) -> float:
        return (
            self.retry_config.initial_delay_seconds
            * self.retry_config.backoff_multiplier ** (attempt - 1)
        )

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
                    messages=messages, model=GROQ_MODEL
                )

                return response.choices[0].message.content
            except AuthenticationError as error:
                logger.exception("Groq authentication failed")

                raise ClientAuthenticationError(
                    "AI client authentication failed"
                ) from error

            except APITimeoutError as error:
                if attempt == max_attempts:
                    logger.exception(
                        "Groq request failed after %d attempts",
                        max_attempts,
                    )

                    raise ClientTimeoutError("AI client request timed out") from error

                delay_seconds = self._calculate_delay(attempt)

                logger.warning(
                    "Groq request timed out on attempt %d/%d. Retrying in %.1f seconds",
                    attempt,
                    max_attempts,
                    delay_seconds,
                )

                time.sleep(delay_seconds)

            except APIConnectionError as error:
                if attempt == max_attempts:
                    logger.exception(
                        "Groq connection failed after %d attempts",
                        max_attempts,
                    )

                    raise ClientConnectionError(
                        "Failed to connect to AI service"
                    ) from error

                delay_seconds = self._calculate_delay(attempt)

                logger.warning(
                    "Groq connection failed on attempt %d/%d. Retrying in %.1f seconds",
                    attempt,
                    max_attempts,
                    delay_seconds,
                )

                time.sleep(delay_seconds)

        raise RuntimeError("Retry loop ended unexpectedly")
