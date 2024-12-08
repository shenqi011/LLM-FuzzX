# src/utils/helpers.py
"""
Helper functions.
"""
import logging
from pathlib import Path
from typing import List
from src.utils.logger import setup_logging

def load_harmful_questions(file_path: str) -> List[str]:
    """
    Loads harmful questions from a file.

    Args:
        file_path (str): Path to the harmful questions file.

    Returns:
        list: List of harmful questions.
    """
    logger = logging.getLogger(__name__)
    
    try:
        path = Path(file_path)
        if not path.exists():
            logger.error(f"Questions file not found: {file_path}")
            raise FileNotFoundError(f"Questions file not found: {file_path}")
            
        with open(path, 'r', encoding='utf-8') as f:
            questions = [line.strip() for line in f if line.strip()]
            
        if not questions:
            logger.warning("No questions found in the file")
            raise ValueError("No questions found in the file")
            
        logger.debug(f"Loaded {len(questions)} questions from {file_path}")
        return questions
        
    except Exception as e:
        logger.error(f"Error loading questions from {file_path}: {e}")
        raise
