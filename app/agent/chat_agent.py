from pathlib import Path
from app.clients.base_client import BaseClient
from app.models.message import Message
from app.models.message_role import MessageRole
from app.memory.base_memory import BaseMemory
from app.config_models.chat_agent_config import ChatAgentConfig
import logging

logger = logging.getLogger(__name__)


class ChatAgent:
    def __init__(
        self,
        config: ChatAgentConfig,
        client: BaseClient,
        memory: BaseMemory,
    ) -> None:
        logger.info("ChatAgent initialized")
        self.app_dir = Path(__file__).resolve().parent.parent
        self.system_prompt = self._load_prompt(config.prompt_name)
        self.system_message = Message(
            role=MessageRole.SYSTEM,
            content=self.system_prompt,
        )
        self.client = client
        self.memory = memory

    def _load_prompt(self, prompt_name: str) -> str:
        prompts_dir = self.app_dir / "prompts"
        file_path = prompts_dir / prompt_name
        return file_path.read_text(encoding="utf-8")

    def _build_messages(self, message: str) -> list[dict[str, str]]:
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

    def chat(self, message: str) -> str:
        logger.info(
            "Processing chat request with message_length=%d",
            len(message),
        )
        messages = self._build_messages(message)
        response = self.client.chat(messages)
        uesr_message = Message(
            role=MessageRole.USER,
            content=message,
        )
        assistant_message = Message(
            role=MessageRole.ASSISTANT,
            content=response,
        )
        self.memory.add_turn(
            user_message=uesr_message,
            assistant_message=assistant_message,
        )
        logger.info("Chat request completed")
        return response
