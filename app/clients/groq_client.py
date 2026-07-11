from groq import Groq
from app.config import GROQ_API_KEY

class GroqClient:
    def __init__(self):
        self.client = Groq(
            api_key=GROQ_API_KEY
        )

    def chat(self, message):
        return "Hello!"