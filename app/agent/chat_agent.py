import logging

from app.clients.base_client import BaseClient
from app.memory.base_memory import BaseMemory
from app.models.message import Message
from app.models.message_role import MessageRole
from app.prompts.prompt_template import PromptTemplate


logger = logging.getLogger(__name__)


class ChatAgent:
    def __init__(
        self,
        prompt_template: PromptTemplate,
        client: BaseClient,
        memory: BaseMemory,
    ) -> None:
        logger.info("ChatAgent initialized")

        self.prompt_template = prompt_template
        self.client = client
        self.memory = memory

        self.system_message = Message(
            role=MessageRole.SYSTEM,
            content=self.prompt_template.render(),
        )

        self.client = client
        self.memory = memory

    def _build_messages(
        self,
        message: str,
    ) -> list[dict[str, str]]:
        logger.debug(
            "Building messages with %d memory items",
            len(self.memory.get_messages()),
        )

        history_messages = [
            history_message.to_dict() for history_message in self.memory.get_messages()
        ]

        messages = [
            self.system_message.to_dict(),
            *history_messages,
            Message(
                role=MessageRole.USER,
                content=message,
            ).to_dict(),
        ]

        return messages

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
