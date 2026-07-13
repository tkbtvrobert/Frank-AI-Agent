from groq import Groq
from app.config import GROQ_API_KEY
from app.config import GROQ_MODEL

class GroqClient:
    def __init__(self):
        self.client = Groq(
            api_key=GROQ_API_KEY
        )

    def chat(self, messages):
        response = self.client.chat.completions.create(messages=messages, model=GROQ_MODEL)
        return response.choices[0].message.content