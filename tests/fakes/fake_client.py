from app.clients.base_client import BaseClient

class FakeClient(BaseClient):
    def chat(self, messages: list[dict[str, str]]) -> str:
        return "This is a fake response."