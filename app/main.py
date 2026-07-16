from app.agent.chat_agent import ChatAgent
from app.clients.groq_client import GroqClient
from app.config_models.chat_agent_config import ChatAgentConfig
from app.memory.sliding_window_memory import SlidingWindowMemory
from app.config_models.memory_config import MemoryConfig

client = GroqClient()

config = ChatAgentConfig(
    prompt_name="system_prompt.txt",
)

memory_config=MemoryConfig(
    max_history_rounds=2,
)

memory = SlidingWindowMemory(
    max_rounds=memory_config.max_history_rounds,
)

agent = ChatAgent(
    config=config,
    client=client,
    memory=memory,
)

print(agent.chat("My name is Frank."))
print(agent.chat("What is my name?"))
print(agent.chat("How are you?"))
print(agent.chat("What are you doing?"))

print(memory.get_messages())
print(len(memory.get_messages()))

memory.clear()

print(memory.get_messages())
