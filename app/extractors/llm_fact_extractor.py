from app.clients.base_client import BaseClient
from app.extractors.base_fact_extractor import BaseFactExtractor
from app.models.message import Message
from app.models.message_role import MessageRole


class LLMFactExtractor(BaseFactExtractor):
    SYSTEM_PROMPT = """
You are a fact extraction assistant.

Extract stable and useful facts about the user from the message.

Return only a valid JSON object.

Rules:
- Use short snake_case keys.
- All values must be strings.
- Do not infer facts that were not explicitly stated.
- Ignore temporary information, questions, opinions, and greetings.
- If there are no useful facts, return {}.

Examples:

User message:
My name is Frank.

Output:
{"user_name": "Frank"}

User message:
I live in Hai Phong.

Output:
{"location": "Hai Phong"}

User message:
How are you?

Output:
{}
""".strip()

    def __init__(
        self,
        client: BaseClient,
    ) -> None:
        self.client = client

    def _build_messages(
        self,
        user_message: str,
    ) -> list[Message]:
        return [
            Message(
                role=MessageRole.SYSTEM,
                content=self.SYSTEM_PROMPT,
            ),
            Message(
                role=MessageRole.USER,
                content=user_message,
            ),
        ]

    def extract(
        self,
        user_message: str,
    ) -> dict[str, str]:
        raise NotImplementedError
