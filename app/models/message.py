from dataclasses import dataclass

from app.models.message_role import MessageRole

@dataclass
class Message:
    role: MessageRole
    content: str