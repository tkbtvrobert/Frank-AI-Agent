from app.agent.chat_agent import ChatAgent
from app.clients.groq_client import GroqClient
from app.config_models.chat_agent_config import ChatAgentConfig

client = GroqClient()

config = ChatAgentConfig(
    prompt_name="system_prompt.txt",
    max_history_rounds=2,
)

agent = ChatAgent(
    config=config,
    client=client,
)

print(agent.chat("My name is Frank."))
print(agent.chat("What is my name?"))
print(agent.chat("How are you?"))
print(agent.chat("What are you doing?"))

print(agent.history)
print(len(agent.history))
