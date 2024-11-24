# src/utils/logger.py
"""
Logger setup and utilities.
"""

import logging

def setup_logging(level='INFO'):
    """
    Sets up the logging configuration.

    Args:
        level (str): Logging level.
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), None),
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
