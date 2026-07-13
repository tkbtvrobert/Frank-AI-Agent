from pathlib import Path

class ChatAgent:
    def __init__(self, prompt_name, client):
        self.app_dir = Path(__file__).resolve().parent.parent
        self.system_prompt = self._load_prompt(prompt_name)
        self.client = client
        self.history = []
        
    def _load_prompt(self, prompt_name):
        prompts_dir = self.app_dir / "prompts"
        file_path = prompts_dir / prompt_name
        return file_path.read_text(encoding='utf-8')
    
    def _build_messages(self, message):
        messages = [
            {
                "role": "system",
                "content": self.system_prompt
            },
            *self.history,
            {
                "role": "user",
                "content": message
            }
        ]
        return messages
    
    def _add_history(self, role, content):
        history = {
            "role": role,
            "content": content
        }
        self.history.append(history)
    
    def chat(self, message):
        messages = self._build_messages(message)
        self._add_history("user", message)
        response = self.client.chat(messages)
        self._add_history("assistant", response)
        return response
