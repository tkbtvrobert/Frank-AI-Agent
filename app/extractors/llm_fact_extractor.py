from app.clients.base_client import BaseClient
from app.extractors.base_fact_extractor import BaseFactExtractor


class LLMFactExtractor(BaseFactExtractor):

    def __init__(
        self,
        client: BaseClient,
    ) -> None:
        self.client = client

    def extract(
        self,
        user_message: str,
    ) -> dict[str, str]:
        raise NotImplementedError