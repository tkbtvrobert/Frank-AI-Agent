from abc import ABC, abstractmethod


class BasePromptTemplate(ABC):
    @abstractmethod
    def render(self) -> str:
        """Render and return the final prompt text."""
        raise NotImplementedError
