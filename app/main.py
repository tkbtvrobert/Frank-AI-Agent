from app.agent.chat_agent import ChatAgent
from app.clients.groq_client import GroqClient
from app.clients.openai_client import OpenAIClient
from tests.fakes.fake_client import FakeClient

client = FakeClient()

agent = ChatAgent(
    prompt_name="system_prompt.txt",
    client=client,
)

# print(agent.chat("My name is Frank."))
# print(agent.chat("What is my name?"))

print(agent.chat("Hello"))
