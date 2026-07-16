from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryConfig:
    max_history_rounds: int = 2
