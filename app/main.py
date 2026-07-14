from app.agent.chat_agent import ChatAgent
from app.clients.groq_client import GroqClient
from app.clients.openai_client import OpenAIClient
from app.models.message import Message
from app.models.message_role import MessageRole

agent = ChatAgent(
    prompt_name="system_prompt.txt",
    client= GroqClient()
)

print(agent.chat("My name is Frank."))
print(agent.chat("What is my name?"))