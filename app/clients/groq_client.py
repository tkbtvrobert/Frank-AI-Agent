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
from groq import (APITimeoutError, AuthenticationError, APIConnectionError)
from app.clients.base_client import BaseClient

class GroqClient(BaseClient):
    def __init__(self) -> None:
        self.client = Groq(
            api_key=GROQ_API_KEY
        )

    def chat(self, messages: list[dict]) -> str:
        try:
            response = self.client.chat.completions.create(messages=messages, model=GROQ_MODEL)
        except AuthenticationError:
            raise
        except APITimeoutError:
            raise
        return response.choices[0].message.content