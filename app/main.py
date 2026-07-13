from app.agent.chat_agent import ChatAgent
from app.clients.groq_client import GroqClient

agent = ChatAgent(
    prompt_name="system_prompt.txt",
    client= GroqClient()
)

print(agent.chat("Hello"))

print(agent.chat("What is Python?"))