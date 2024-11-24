# src/models/openai_model.py
"""
OpenAIModel class to interact with OpenAI API.
"""

import openai
from config import OPENAI_API_KEY

class OpenAIModel:
    """
    Class to interact with OpenAI models via API.
    """

    def __init__(self, model_name, **kwargs):
        """
        Initializes the OpenAIModel.

        Args:
            model_name (str): Name of the OpenAI model to use.
            **kwargs: Additional arguments.
        """
        openai.api_key = OPENAI_API_KEY
        self.model_name = model_name

    def generate(self, prompt, max_tokens=512, temperature=0.7, **kwargs):
        """
        Generates a response using the OpenAI API.

        Args:
            prompt (str): The input prompt.
            max_tokens (int): Maximum number of tokens to generate.
            temperature (float): Sampling temperature.
            **kwargs: Additional arguments.

        Returns:
            str: The generated response.
        """
        response = openai.Completion.create(
            engine=self.model_name,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
        return response['choices'][0]['text'].strip()
