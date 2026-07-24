from app.policies.base_memory_policy import BaseMemoryPolicy


class FakeMemoryPolicy(BaseMemoryPolicy):
    def __init__(self, should_remember_result: bool):
        self.should_remember_result = should_remember_result

    def should_remember(
        self,
        key: str,
        value: str,
    ) -> bool:
        self.received_key = key
        self.received_value = value
        return self.should_remember_result
