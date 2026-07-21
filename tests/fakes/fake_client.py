from app.clients.base_client import BaseClient
from app.models.message import Message


class FakeClient(BaseClient):
    def chat(
        self,
        messages: list[Message],
    ) -> str:
        return "This is a fake response."
