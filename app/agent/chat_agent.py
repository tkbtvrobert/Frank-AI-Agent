import logging

from app.clients.base_client import BaseClient
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
    ) -> None:
        logger.info("ChatAgent initialized")

        self.prompt_template = prompt_template
        self.client = client
        self.memory = memory

        rendered_prompt = self.prompt_template.render()

        if not rendered_prompt.strip():
            raise ValueError("Rendered system prompt cannot be empty")

        self.system_message = Message(
            role=MessageRole.SYSTEM,
            content=rendered_prompt,
        )

    def _build_messages(
        self,
        message: str,
    ) -> list[Message]:
        history_messages = self.memory.get_messages()

        logger.debug(
            "Building messages with %d memory items",
            len(history_messages),
        )

        return [
            self.system_message,
            *history_messages,
            Message(
                role=MessageRole.USER,
                content=message,
            ),
        ]

    def chat(
        self,
        message: str,
    ) -> str:
        logger.info(
            "Processing chat request with message_length=%d",
            len(message),
        )

        messages = self._build_messages(message)
        response = self.client.chat(messages)

        user_message = Message(
            role=MessageRole.USER,
            content=message,
        )

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
