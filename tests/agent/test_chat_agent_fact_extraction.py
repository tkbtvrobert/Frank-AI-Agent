import pytest

from app.agent.chat_agent import ChatAgent
from app.config_models.memory_config import MemoryConfig
from app.config_models.prompt_config import PromptConfig
from app.extractors.regex_fact_extractor import RegexFactExtractor
from app.memory.in_memory_fact_memory import InMemoryFactMemory
from app.memory.sliding_window_memory import SlidingWindowMemory
from app.models.message import Message
from app.models.message_role import MessageRole
from app.prompts.prompt_template import PromptTemplate
from tests.fakes.fake_client import FakeClient


@pytest.fixture
def fake_client() -> FakeClient:
    return FakeClient(
        response="測試回覆",
    )


@pytest.fixture
def agent(
    fake_client: FakeClient,
) -> ChatAgent:
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
        client=fake_client,
        memory=memory,
        fact_memory=InMemoryFactMemory(),
        fact_extractor=RegexFactExtractor(),
    )


def test_chat_automatically_remembers_extracted_fact(
    agent: ChatAgent,
) -> None:
    agent.chat("My name is Frank.")

    assert agent.get_fact("user_name") == "Frank"


def test_chat_does_not_create_fact_when_none_is_extracted(
    agent: ChatAgent,
) -> None:
    agent.chat("How are you?")

    assert agent.get_fact("user_name") is None


def test_extracted_fact_is_injected_into_system_message(
    agent: ChatAgent,
    fake_client: FakeClient,
) -> None:
    agent.chat("My name is Frank.")

    assert fake_client.call_count == 1

    system_message = fake_client.received_messages[0]

    assert system_message.role == MessageRole.SYSTEM
    assert "Known user facts:" in system_message.content
    assert "- user_name: Frank" in system_message.content


def test_chat_stores_conversation_turn(
    agent: ChatAgent,
) -> None:
    agent.chat("My name is Frank.")

    messages = agent.memory.get_messages()

    assert messages == [
        Message(
            role=MessageRole.USER,
            content="My name is Frank.",
        ),
        Message(
            role=MessageRole.ASSISTANT,
            content="測試回覆",
        ),
    ]


def test_new_extracted_fact_overwrites_existing_fact(
    agent: ChatAgent,
) -> None:
    agent.chat("My name is Frank.")
    agent.chat("My name is David.")

    assert agent.get_fact("user_name") == "David"