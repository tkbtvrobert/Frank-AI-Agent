from app.agent.chat_agent import ChatAgent
from app.clients.groq_client import GroqClient

agent = ChatAgent()

client = GroqClient()

print("Client Ready")

print(client.chat("Hello"))

print("Frank AI Agent")

print(agent._load_prompt("system_prompt.txt"))

print(agent._load_prompt("coding_prompt.txt"))