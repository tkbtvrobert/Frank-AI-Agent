from pathlib import Path
from app.clients.base_client import BaseClient
from typing import Any
from app.models.message_role import MessageRole

class ChatAgent:
    def __init__(self, prompt_name: str, client: BaseClient) -> None:
        self.app_dir = Path(__file__).resolve().parent.parent
        self.system_prompt = self._load_prompt(prompt_name)
        self.client = client
        self.history: list[dict[str, Any]] = []
        
    def _load_prompt(self, prompt_name: str) -> str:
        prompts_dir = self.app_dir / "prompts"
        file_path = prompts_dir / prompt_name
        return file_path.read_text(encoding='utf-8')
    
    def _build_messages(self, message:str) -> list[dict[str, Any]]:
        messages = [
            {
                "role": MessageRole.SYSTEM.value,
                "content": self.system_prompt
            },
            *self.history,
            {
                "role": MessageRole.USER.value,
                "content": message
            }
        ]
        return messages
    
    def _add_history(self, role: str, content: str) -> None:
        history = {
            "role": role,
            "content": content
        }
        self.history.append(history)
    
    def chat(self, message: str) -> str:
        messages = self._build_messages(message)
        self._add_history(MessageRole.USER.value, message)
        response = self.client.chat(messages)
        self._add_history(MessageRole.ASSISTANT.value, response)
        return response
