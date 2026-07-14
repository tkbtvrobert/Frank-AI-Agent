from pathlib import Path
from app.clients.base_client import BaseClient
from app.models.message import Message
from app.models.message_role import MessageRole

class ChatAgent:
    def __init__(self, prompt_name: str, client: BaseClient) -> None:
        self.app_dir = Path(__file__).resolve().parent.parent
        self.system_prompt = self._load_prompt(prompt_name)
        self.client = client
        self.history: list[Message] = []
        
    def _load_prompt(self, prompt_name: str) -> str:
        prompts_dir = self.app_dir / "prompts"
        file_path = prompts_dir / prompt_name
        return file_path.read_text(encoding='utf-8')
    
    def _build_messages(self, message:str) -> list[dict[str, str]]:
        history_messages = [
            history_message.to_dict()
            for history_message in self.history
        ]
        messages = [
            Message(
                role=MessageRole.SYSTEM,
                content=self.system_prompt,
            ).to_dict(),
            *history_messages,
            Message(
                role=MessageRole.USER,
                content=message,
            ).to_dict(),
        ]
        return messages
    
    def _add_history(self, role: MessageRole, content: str) -> None:
        history_message = Message(
            role=role,
            content=content,
        )
        self.history.append(history_message)
    
    def chat(self, message: str) -> str:
        messages = self._build_messages(message)
        self._add_history(MessageRole.USER, message)
        response = self.client.chat(messages)
        self._add_history(MessageRole.ASSISTANT, response)
        return response
