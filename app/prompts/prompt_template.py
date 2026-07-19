from pathlib import Path
from typing import Any


class PromptTemplate:
    def __init__(
        self,
        prompt_name: str,
    ) -> None:
        self.prompt_name = prompt_name

    def _get_prompt_path(self) -> Path:
        return Path(__file__).parent / self.prompt_name

    def _load_template(self) -> str:
        prompt_path = self._get_prompt_path()

        print(prompt_path)
        print(prompt_path.exists())

        return prompt_path.read_text(
            encoding="utf-8",
        )

    def render(
        self,
        **variables: Any,
    ) -> str:
        template = self._load_template()

        return template.format(**variables)
