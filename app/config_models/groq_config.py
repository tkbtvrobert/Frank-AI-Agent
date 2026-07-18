from dataclasses import dataclass


@dataclass(frozen=True)
class GroqConfig:
    api_key: str
    model: str