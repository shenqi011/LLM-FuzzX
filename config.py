# config.py
"""
Configuration file for project constants and settings.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API keys (should be set via environment variables for security)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
CLAUDE_API_KEY = os.getenv('CLAUDE_API_KEY')

# OpenAI API configuration
BASE_URL = os.getenv('OPENAI_API_ENDPOINT', 'https://api.deepseek.com')

MAX_TOKENS = 512

# Logging level
LOGGING_LEVEL = 'INFO'

# Bing Translator API Configuration
BING_TRANSLATE_API_KEY = "9fdvcdqxLLxVlmugL6eWTzsLYGIZxNeUcLxmy6FMkwjkJ2az1actJQQJ99BEAC3pKaRXJ3w3AAAbACOG9e6f"
BING_TRANSLATE_LOCATION = "eastasia"  # Or your specific region like "eastus" 

MODEL_CONFIG = {
    'target_model': 'deepseek-chat',
    'mutator_model': 'deepseek-chat',
    'evaluator_model': 'roberta-base',
    'temperature': 0.7,
    'max_tokens': 2048
}