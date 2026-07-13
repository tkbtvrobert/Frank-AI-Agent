from pathlib import Path

class ChatAgent:
    def __init__(self, prompt_name, client):
        self.app_dir = Path(__file__).resolve().parent.parent
        self.system_prompt = self._load_prompt(prompt_name)
        self.client = client
        
    def _load_prompt(self, prompt_name):
        prompts_dir = self.app_dir / "prompts"
        file_path = prompts_dir / prompt_name
        return file_path.read_text(encoding='utf-8')
    
    def _build_messages(self, messages):
        messages = [
            {
                "role": "system",
                "content": self.system_prompt
            },
            {
                "role": "user",
                "content": messages
            }
        ]
        return messages
    
    def chat(self, messages):
        messages = self._build_messages(messages)
        return self.client.chat(messages)
