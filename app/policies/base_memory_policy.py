from abc import ABC, abstractmethod


class BaseMemoryPolicy(ABC):
    @abstractmethod
    def should_remember(
        self,
        key: str,
        value: str,
    ) -> bool:
        raise NotImplementedError
