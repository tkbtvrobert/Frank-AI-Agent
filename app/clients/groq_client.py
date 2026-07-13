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

class GroqClient:
    def __init__(self):
        self.client = Groq(
            api_key=GROQ_API_KEY
        )

    def chat(self, messages):
        try:
            response = self.client.chat.completions.create(messages=messages, model=GROQ_MODEL)
        except AuthenticationError as e:
            raise
        except APITimeoutError as e:
            raise
        return response.choices[0].message.content