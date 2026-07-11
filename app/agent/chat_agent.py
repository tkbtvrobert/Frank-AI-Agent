from pathlib import Path

class ChatAgent:
    def __init__(self):
        self.app_dir = Path(__file__).resolve().parent.parent
        self.prompts_dir = self.app_dir / "prompts"
    
    def _load_prompt(self, filename):
        file_path = self.prompts_dir / filename
        return file_path.read_text(encoding='utf-8')