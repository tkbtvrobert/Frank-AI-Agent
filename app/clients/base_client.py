from abc import ABC, abstractmethod
from app.models.message import Message


class BaseClient(ABC):
    @abstractmethod
    def chat(self, messages: list[Message]) -> str:
        pass
