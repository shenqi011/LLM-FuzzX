# src/utils/helpers.py
"""
Helper functions.
"""

def load_harmful_questions(file_path='data/questions/harmful_questions.txt'):
    """
    Loads harmful questions from a file.

    Args:
        file_path (str): Path to the harmful questions file.

    Returns:
        list: List of harmful questions.
    """
    with open(file_path, 'r') as f:
        questions = [line.strip() for line in f if line.strip()]
    return questions
