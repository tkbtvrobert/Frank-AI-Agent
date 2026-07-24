import pytest

from app.agent.chat_agent import ChatAgent
from app.config_models.memory_config import MemoryConfig
from app.config_models.prompt_config import PromptConfig
from app.extractors.regex_fact_extractor import RegexFactExtractor
from app.memory.in_memory_fact_memory import InMemoryFactMemory
from app.memory.sliding_window_memory import SlidingWindowMemory
from app.models.message import Message
from app.models.message_role import MessageRole
from app.policies.simple_memory_policy import SimpleMemoryPolicy
from app.prompts.prompt_template import PromptTemplate
from tests.fakes.fake_client import FakeClient
from tests.fakes.fake_memory_policy import FakeMemoryPolicy


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
        memory_policy=SimpleMemoryPolicy(),
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


def test_chat_agent_stores_memory_policy() -> None:
    prompt_config = PromptConfig(
        prompt_name="system_prompt.txt",
        user_name="Frank",
        language="Traditional Chinese",
    )

    memory_policy = SimpleMemoryPolicy()

    agent = ChatAgent(
        prompt_template=PromptTemplate(
            config=prompt_config,
        ),
        client=FakeClient(),
        memory=SlidingWindowMemory(
            max_rounds=10,
        ),
        fact_memory=InMemoryFactMemory(),
        fact_extractor=RegexFactExtractor(),
        memory_policy=memory_policy,
    )

    assert agent.memory_policy is memory_policy


def test_chat_does_not_store_fact_when_memory_policy_rejects_it() -> None:
    prompt_config = PromptConfig(
        prompt_name="system_prompt.txt",
        user_name="Frank",
        language="Traditional Chinese",
    )

    memory_policy = FakeMemoryPolicy(
        should_remember_result=False,
    )

    agent = ChatAgent(
        prompt_template=PromptTemplate(
            config=prompt_config,
        ),
        client=FakeClient(),
        memory=SlidingWindowMemory(
            max_rounds=10,
        ),
        fact_memory=InMemoryFactMemory(),
        fact_extractor=RegexFactExtractor(),
        memory_policy=memory_policy,
    )

    agent.chat("My name is Frank.")

    assert agent.get_fact("user_name") is None


def test_chat_stores_fact_when_memory_policy_allows_it() -> None:
    prompt_config = PromptConfig(
        prompt_name="system_prompt.txt",
        user_name="Frank",
        language="Traditional Chinese",
    )

    memory_policy = FakeMemoryPolicy(
        should_remember_result=True,
    )

    agent = ChatAgent(
        prompt_template=PromptTemplate(
            config=prompt_config,
        ),
        client=FakeClient(),
        memory=SlidingWindowMemory(
            max_rounds=10,
        ),
        fact_memory=InMemoryFactMemory(),
        fact_extractor=RegexFactExtractor(),
        memory_policy=memory_policy,
    )

    agent.chat("My name is Frank.")

    assert agent.get_fact("user_name") == "Frank"
    assert memory_policy.received_key == "user_name"
    assert memory_policy.received_value == "Frank"
