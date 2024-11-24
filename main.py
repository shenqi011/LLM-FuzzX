# main.py
"""
Main script to run the fuzzing process.
"""

import argparse
from src.models.llm_wrapper import LLMWrapper
from src.fuzzing.fuzzing_engine import FuzzingEngine
from src.fuzzing.seed_selector import TargetAwareSeedSelector
from src.fuzzing.mutator import TargetAwareMutator
from src.fuzzing.evaluator import ResponseEvaluator
from src.utils.logger import setup_logging

def parse_arguments():
    parser = argparse.ArgumentParser(description='Fuzzing LLMs to generate jailbreak prompts.')
    parser.add_argument('--model_name', type=str, default='llama2-7b-chat', help='Target model name.')
    parser.add_argument('--max_iterations', type=int, default=1000, help='Maximum number of iterations.')
    # Add other arguments as needed
    return parser.parse_args()

def main():
    args = parse_arguments()
    setup_logging()

    # Initialize LLM
    target_model = LLMWrapper(model_name=args.model_name)

    # Initialize components
    seed_selector = TargetAwareSeedSelector(target_model)
    mutator = TargetAwareMutator(target_model)
    evaluator = ResponseEvaluator()

    # Initialize fuzzing engine
    fuzzing_engine = FuzzingEngine(
        target_model=target_model,
        seed_selector=seed_selector,
        mutator=mutator,
        evaluator=evaluator,
        max_iterations=args.max_iterations
    )

    # Run fuzzing process
    fuzzing_engine.run()

if __name__ == '__main__':
    main()
