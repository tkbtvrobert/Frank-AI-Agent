from dataclasses import dataclass
from app.models.message_role import MessageRole

@dataclass
class Message:
    role: MessageRole
    content: str

    def to_dict(self) -> dict[str, str]:
        return {
            "role": self.role.value,
            "content": self.content,
        }