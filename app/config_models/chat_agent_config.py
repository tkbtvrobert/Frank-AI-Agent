from dataclasses import dataclass


@dataclass(frozen=True)
class ChatAgentConfig:
    prompt_name: str
    max_history_rounds: int = 2
