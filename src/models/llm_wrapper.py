# src/models/llm_wrapper.py
"""
LLMWrapper class that provides a unified interface to interact with different LLMs.
"""

from src.models.openai_model import OpenAIModel
from src.models.claude_model import ClaudeModel
from src.models.llama_model import LlamaModel

class LLMWrapper:
    """
    Wrapper class for various LLMs.
    """

    def __init__(self, model_name, **kwargs):
        """
        Initializes the LLMWrapper with the specified model.

        Args:
            model_name (str): Name of the model to use.
            **kwargs: Additional arguments for model initialization.
        """
        self.model_name = model_name
        if 'openai' in model_name.lower():
            self.model = OpenAIModel(model_name, **kwargs)
        elif 'claude' in model_name.lower():
            self.model = ClaudeModel(model_name, **kwargs)
        elif 'llama' in model_name.lower():
            self.model = LlamaModel(model_name, **kwargs)
        else:
            raise ValueError(f'Unsupported model: {model_name}')

    def generate(self, prompt, **kwargs):
        """
        Generates a response from the model given a prompt.

        Args:
            prompt (str): The input prompt.
            **kwargs: Additional arguments for generation.

        Returns:
            str: The generated response.
        """
        return self.model.generate(prompt, **kwargs)
