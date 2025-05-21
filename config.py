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
TRANSLATE_API_KEY = "not_use"

# OpenAI API configuration
BASE_URL = os.getenv('OPENAI_API_ENDPOINT', 'https://api.deepseek.com')

MAX_TOKENS = 512

# Logging level
LOGGING_LEVEL = 'INFO'

MODEL_CONFIG = {
    'target_model': 'deepseek-chat',
    'mutator_model': 'deepseek-chat',
    'evaluator_model': 'roberta-base',
    'temperature': 0.7,
    'max_tokens': 2048
}