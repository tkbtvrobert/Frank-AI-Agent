from pathlib import Path
from typing import Any

from app.config_models.prompt_config import PromptConfig


class PromptTemplate:
    def __init__(
        self,
        config: PromptConfig,
    ) -> None:
        self.config = config

    def _get_prompt_path(self) -> Path:
        return Path(__file__).parent / self.config.prompt_name

    def _load_template(self) -> str:
        prompt_path = self._get_prompt_path()

        return prompt_path.read_text(
            encoding="utf-8",
        )

    def render(self) -> str:
        template = self._load_template()

        return template.format(
            user_name=self.config.user_name,
            language=self.config.language,
        )
