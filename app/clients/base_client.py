from abc import ABC, abstractmethod
class BaseClient(ABC):

    @abstractmethod
    def chat(self, messages: list[dict]) -> str:
        pass