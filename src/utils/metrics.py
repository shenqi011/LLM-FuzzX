# src/utils/metrics.py
"""
Metrics calculation functions.
"""

def calculate_attack_success_rate(results):
    """
    Calculates the attack success rate.

    Args:
        results (list): List of evaluation results.

    Returns:
        float: Success rate.
    """
    total = len(results)
    successes = sum(1 for result in results if result['is_successful'])
    return successes / total if total > 0 else 0.0
