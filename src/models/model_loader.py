from config import config
from models.api_models import generate_api_response, load_api_model
from models.local_models import generate_local_response, load_local_model
from models.prompt_utils import load_messages

CONFIG = config["model_config"]


class ModelWrapper:
    def __init__(self, model_name):
        self.model_name = model_name
        self.is_api_model = model_name in [
            "claude-3-opus",
            "claude-3-5-sonnet",
            "deepseek-v2",
            "gemini-1.5-pro",
            "gpt-4",
            "gpt-4o",
            "gpt-4o-mini",
        ]

        if self.is_api_model:
            self.model = load_api_model(model_name)
        else:
            self.model, self.tokenizer = load_local_model(model_name)

    def generate_response(self, prompt):
        messages = load_messages(self.model_name, prompt)
        if self.is_api_model:
            return generate_api_response(self.model_name, self.model, messages)
        else:
            return generate_local_response(
                self.model_name, self.model, self.tokenizer, messages
            )
