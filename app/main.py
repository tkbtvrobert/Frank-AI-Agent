from app.agent.chat_agent import ChatAgent
from app.clients.groq_client import GroqClient
from app.config import GROQ_API_KEY, GROQ_MODEL
from app.config_models.groq_config import GroqConfig
from app.config_models.memory_config import MemoryConfig
from app.config_models.prompt_config import PromptConfig
from app.config_models.retry_config import RetryConfig
from app.core.logging_config import configure_logging
from app.exceptions.client_exceptions import (
    AIClientError,
    ClientAuthenticationError,
    ClientConnectionError,
    ClientTimeoutError,
)
from app.memory.sliding_window_memory import SlidingWindowMemory
from app.prompts.prompt_template import PromptTemplate


def run_demo(agent: ChatAgent) -> None:
    messages = [
        "My name is Frank.",
        "What is my name?",
        "How are you?",
        "What are you doing?",
    ]

    for message in messages:
        response = agent.chat(message)
        print(response)

    print(agent.memory.get_messages())
    print(len(agent.memory.get_messages()))

    agent.memory.clear()
    print(agent.memory.get_messages())


def create_agent() -> ChatAgent:
    groq_config = GroqConfig(
        api_key=GROQ_API_KEY,
        model=GROQ_MODEL,
    )
    retry_config = RetryConfig(
        max_attempts=3,
        initial_delay_seconds=1,
        backoff_multiplier=2.0,
    )
    client = GroqClient(
        groq_config=groq_config,
        retry_config=retry_config,
    )

    prompt_config = PromptConfig(
        prompt_name="system_prompt.txt",
        user_name="Frank",
        language="Traditional Chinese",
    )

    prompt_template = PromptTemplate(
        config=prompt_config,
    )

    memory_config = MemoryConfig(
        max_history_rounds=2,
    )

    memory = SlidingWindowMemory(
        max_rounds=memory_config.max_history_rounds,
    )

    return ChatAgent(
        prompt_template=prompt_template,
        client=client,
        memory=memory,
    )


def main() -> None:
    configure_logging()

    try:
        agent = create_agent()
        run_demo(agent)
    except ClientAuthenticationError:
        print("Authentication failed. Please check the API configuration.")

    except ClientTimeoutError:
        print("The AI service took too long to respond.")

    except ClientConnectionError:
        print("Unable to connect to the AI service.")

    except AIClientError as error:
        print(f"AI service error: {error}")


if __name__ == "__main__":
    main()
