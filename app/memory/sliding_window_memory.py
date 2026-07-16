from app.memory.base_memory import BaseMemory
from app.models.message import Message


class SlidingWindowMemory(BaseMemory):
    def __init__(self, max_rounds: int = 2) -> None:
        self.max_rounds = max_rounds
        self.messages: list[Message] = []

    def add_turn(self, user_message, assistant_message):
        self.messages.extend(
            [
                user_message,
                assistant_message,
            ]
        )

        max_messages = self.max_rounds * 2

        if len(self.messages) > max_messages:
            self.messages = self.messages[-max_messages:]

    def get_messages(self) -> list[Message]:
        return self.messages
    
    def clear(self) -> None:
        return self.messages.clear()