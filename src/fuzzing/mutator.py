# src/fuzzing/mutator.py
"""
Mutator classes for mutation strategies.
"""

import random

class Mutator:
    """
    Base class for mutation strategies.
    """

    def __init__(self, **kwargs):
        """
        Initializes the Mutator.

        Args:
            **kwargs: Additional arguments.
        """
        pass

    def mutate(self, seed):
        """
        Performs mutation on the seed prompt.

        Args:
            seed (str): The seed prompt to mutate.

        Returns:
            str: The mutated prompt.
        """
        return seed  # No mutation by default

class TargetAwareMutator(Mutator):
    """
    Implements target-aware mutation strategies.
    """

    def __init__(self, target_model, **kwargs):
        """
        Initializes the TargetAwareMutator.

        Args:
            target_model: The target LLM model.
            **kwargs: Additional arguments.
        """
        super().__init__(**kwargs)
        self.target_model = target_model

    def mutate(self, seed):
        """
        Implements the target-aware mutation logic.

        Args:
            seed (str): The seed prompt to mutate.

        Returns:
            str: The mutated prompt.
        """
        # Implement target-aware mutation logic
        # Example mutations: synonym replacement, grammatical transformations, etc.
        mutations = [
            lambda s: s.replace(' and ', ' & '),
            lambda s: s.replace('your', 'ur'),
            lambda s: s[::-1],  # Reverse the string
            lambda s: s.upper(),
            lambda s: s.lower()
        ]
        mutation = random.choice(mutations)
        mutated_seed = mutation(seed)
        return mutated_seed
