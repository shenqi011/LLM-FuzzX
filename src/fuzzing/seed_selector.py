# src/fuzzing/seed_selector.py
"""
SeedSelector classes for seed selection strategies.
"""

import random

class SeedSelector:
    """
    Base class for seed selection strategies.
    """

    def __init__(self, seed_pool=None, **kwargs):
        """
        Initializes the SeedSelector.

        Args:
            seed_pool (list): List of seed prompts.
            **kwargs: Additional arguments.
        """
        if seed_pool is None:
            # Load default seeds from data/seeds/initial_seeds.txt
            with open('data/seeds/initial_seeds.txt', 'r') as f:
                self.seed_pool = [line.strip() for line in f if line.strip()]
        else:
            self.seed_pool = seed_pool

    def select_seed(self):
        """
        Selects a seed prompt.

        Returns:
            str: The selected seed prompt.
        """
        return random.choice(self.seed_pool)

class TargetAwareSeedSelector(SeedSelector):
    """
    Implements target-aware seed selection.
    """

    def __init__(self, target_model, **kwargs):
        """
        Initializes the TargetAwareSeedSelector.

        Args:
            target_model: The target LLM model.
            **kwargs: Additional arguments.
        """
        super().__init__(**kwargs)
        self.target_model = target_model

    def select_seed(self):
        """
        Implements the target-aware seed selection logic.

        Returns:
            str: The selected seed prompt.
        """
        # Implement target-aware seed selection logic
        # For example, analyze past responses to prioritize certain seeds
        seed = super().select_seed()
        # Placeholder: Modify seed selection based on target model characteristics
        return seed
