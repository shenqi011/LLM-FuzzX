# src/models/llama_model.py
"""
LlamaModel class to interact with LLaMA models.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class LlamaModel:
    """
    Class to interact with LLaMA models.
    """

    def __init__(self, model_name, device='cuda', precision='float16', **kwargs):
        """
        Initializes the LlamaModel.

        Args:
            model_name (str): Name or path of the LLaMA model.
            device (str): Device to load the model on.
            precision (str): Precision of the model (e.g., 'float16').
            **kwargs: Additional arguments.
        """
        self.model_name = model_name
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=getattr(torch, precision)
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()

    def generate(self, prompt, max_new_tokens=512, temperature=0.7, **kwargs):
        """
        Generates a response using the LLaMA model.

        Args:
            prompt (str): The input prompt.
            max_new_tokens (int): Maximum number of new tokens to generate.
            temperature (float): Sampling temperature.
            **kwargs: Additional arguments.

        Returns:
            str: The generated response.
        """
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
