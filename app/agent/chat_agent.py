import logging

from app.clients.base_client import BaseClient
from app.extractors.base_fact_extractor import BaseFactExtractor
from app.memory.base_fact_memory import BaseFactMemory
from app.memory.base_memory import BaseMemory
from app.models.message import Message
from app.models.message_role import MessageRole
from app.prompts.base_prompt_template import BasePromptTemplate


logger = logging.getLogger(__name__)


class ChatAgent:
    def __init__(
        self,
        prompt_template: BasePromptTemplate,
        client: BaseClient,
        memory: BaseMemory,
        fact_memory: BaseFactMemory,
        fact_extractor: BaseFactExtractor,
    ) -> None:
        logger.info("ChatAgent initialized")

        self.prompt_template = prompt_template
        self.client = client
        self.memory = memory
        self.fact_memory = fact_memory
        self.fact_extractor = fact_extractor

        rendered_prompt = self.prompt_template.render()

        if not rendered_prompt.strip():
            raise ValueError("Rendered system prompt cannot be empty")

        self.system_message = Message(
            role=MessageRole.SYSTEM,
            content=rendered_prompt,
        )

    def _remember_extracted_facts(
        self,
        facts: dict[str, str],
    ) -> None:
        if not facts:
            logger.debug("No facts extracted from user message")
            return

        for key, value in facts.items():
            self.fact_memory.set(
                key,
                value,
            )

            logger.debug(
                "Remembered extracted fact key=%s value=%r",
                key,
                value,
            )

    def _format_facts(
        self,
        facts: dict[str, str],
    ) -> str | None:
        if not facts:
            return None

        formatted_facts = "\n".join(f"- {key}: {value}" for key, value in facts.items())

        return f"Known user facts:\n{formatted_facts}"

    def _build_messages(
        self,
        user_message: Message,
    ) -> list[Message]:
        history_messages = self.memory.get_messages()
        facts = self.fact_memory.get_all()

        system_content = self.system_message.content
        facts_content = self._format_facts(facts)

        if facts_content is not None:
            system_content = f"{system_content}\n\n{facts_content}"

        system_message = Message(
            role=MessageRole.SYSTEM,
            content=system_content,
        )

        logger.debug(
            "Building messages with %d memory items and %d facts",
            len(history_messages),
            len(facts),
        )

        return [
            system_message,
            *history_messages,
            user_message,
        ]

    def remember_fact(
        self,
        key: str,
        value: str,
    ) -> None:
        self.fact_memory.set(key, value)

    def get_fact(
        self,
        key: str,
    ) -> str | None:
        return self.fact_memory.get(key)

    def forget_fact(
        self,
        key: str,
    ) -> None:
        self.fact_memory.delete(key)

    def chat(
        self,
        message: str,
    ) -> str:
        if not message.strip():
            raise ValueError("User message cannot be empty")

        logger.info(
            "Processing chat request with message_length=%d",
            len(message),
        )

        user_message = Message(
            role=MessageRole.USER,
            content=message,
        )

        facts = self.fact_extractor.extract(
            user_message.content,
        )

        self._remember_extracted_facts(
            facts,
        )

        messages = self._build_messages(user_message)
        response = self.client.chat(messages)

        assistant_message = Message(
            role=MessageRole.ASSISTANT,
            content=response,
        )

        self.memory.add_turn(
            user_message=user_message,
            assistant_message=assistant_message,
        )

        logger.info("Chat request completed")

        return response
