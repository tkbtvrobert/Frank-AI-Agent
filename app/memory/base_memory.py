from abc import ABC, abstractmethod
from app.models.message import Message


class BaseMemory(ABC):
    @abstractmethod
    def add_turn(self, user_message: Message, assistant_message: Message) -> None:
        pass

    @abstractmethod
    def get_messages(self) -> list[Message]:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass
