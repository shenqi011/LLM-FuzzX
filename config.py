# config.py
"""
Configuration file for project constants and settings.
"""

import os

# API keys (should be set via environment variables for security)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
CLAUDE_API_KEY = os.getenv('CLAUDE_API_KEY')
OPENAI_API_ENDPOINT = os.getenv('OPENAI_API_ENDPOINT', 'https://api.openai.com')
CLAUDE_API_ENDPOINT = os.getenv('CLAUDE_API_ENDPOINT', 'https://claude-api.com')

# Model settings
DEFAULT_MODEL_NAME = 'llama2-7b-chat'
MAX_TOKENS = 512

# Logging level
LOGGING_LEVEL = 'INFO'
