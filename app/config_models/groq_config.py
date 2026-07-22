from dataclasses import dataclass


@dataclass(frozen=True)
class GroqConfig:
    api_key: str
    model: str
    temperature: float = 0.7
    max_tokens: int | None = 256