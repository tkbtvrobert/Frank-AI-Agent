from dataclasses import dataclass


@dataclass
class RetryConfig:
    max_attempts: int = 3
    initial_delay_seconds: float = 1.0