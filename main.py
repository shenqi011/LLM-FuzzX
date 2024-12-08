# main.py
"""
Main script to run the fuzzing process.
"""

import argparse
import logging
from pathlib import Path
import pandas as pd
from datetime import datetime
import json

from src.models.openai_model import OpenAIModel
from src.fuzzing.fuzzing_engine import FuzzingEngine
from src.fuzzing.seed_selector import UCBSeedSelector
from src.fuzzing.mutator import LLMMutator
from src.evaluation.roberta_evaluator import RoBERTaEvaluator
from src.utils.logger import setup_logging
from src.utils.helpers import load_harmful_questions

def parse_arguments():
    parser = argparse.ArgumentParser(description='Fuzzing LLMs to generate jailbreak prompts.')
    
    # 模型相关参数
    parser.add_argument('--target_model', type=str, default='gpt-3.5-turbo', 
                        help='Target model name')
    parser.add_argument('--mutator_model', type=str, default='gpt-4o',
                        help='Model used for mutation')
    
    # 数据相关参数
    parser.add_argument('--seed_path', type=str, 
                        default='data/seeds/GPTFuzzer.csv',
                        help='Path to seed prompts')
    parser.add_argument('--questions_path', type=str,
                        default='data/questions/harmful_questions.txt',
                        help='Path to harmful questions')
    
    # Fuzzing参数
    parser.add_argument('--max_iterations', type=int, default=5,
                        help='Maximum number of iterations')
    parser.add_argument('--exploration_weight', type=float, default=1.0,
                        help='Exploration weight for UCB')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Temperature for LLM mutation')
    
    # 输出相关参数
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--log_level', type=str, default='DEBUG',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    
    return parser.parse_args()

def setup_output_dir(output_dir: str) -> Path:
    """设置输出目录"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir) / timestamp
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path

def log_dict(logger: logging.Logger, data: dict, level: int = logging.INFO):
    """格式化记录字典数据"""
    logger.log(level, json.dumps(data, indent=2, ensure_ascii=False))

def setup_multi_logger(log_dir: Path, level: str = 'INFO') -> dict:
    """
    设置分层日志系统
    
    Args:
        log_dir: 日志目录路径
        level: 日志级别
        
    Returns:
        dict: 包含各个logger的字典
    """
    # 确保日志目录存在
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 基础日志格式
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 定义不同的logger和对应的文件
    loggers = {
        'main': {
            'file': log_dir / 'main.log',
            'level': level
        },
        'api': {
            'file': log_dir / 'api.log',
            'level': level
        },
        'mutation': {
            'file': log_dir / 'mutation.log',
            'level': level
        },
        'stats': {
            'file': log_dir / 'stats.log',
            'level': level
        }
    }
    
    # 创建和配置每个logger
    logger_dict = {}
    for name, config in loggers.items():
        # 创建logger
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, config['level']))
        logger.propagate = False  # 防止日志传播到父logger
        
        # 添加文件处理器
        file_handler = logging.FileHandler(config['file'], encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(getattr(logging, config['level']))
        logger.addHandler(file_handler)
        
        # 添加控制台处理器(仅对main logger)
        if name == 'main':
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            console_handler.setLevel(getattr(logging, level))
            logger.addHandler(console_handler)
            
        logger_dict[name] = logger
    
    return logger_dict

def main():
    # 解析参数
    args = parse_arguments()
    
    # 设置输出目录
    output_path = setup_output_dir(args.output_dir)
    
    # 设置日志
    loggers = setup_multi_logger(
        level=args.log_level,
        log_dir=output_path
    )
    
    try:
        # 加载种子和问题
        initial_seeds = load_seeds(args.seed_path)
        harmful_questions = load_harmful_questions(args.questions_path)
        
        # 记录初始状态
        loggers['api'].info(json.dumps({
            "initial_seeds": len(initial_seeds),
            "harmful_questions": len(harmful_questions),
            "target_model": args.target_model,
            "mutator_model": args.mutator_model
        }))
        
        # 初始化模型
        target_model = OpenAIModel(model_name=args.target_model)
        mutator_model = OpenAIModel(model_name=args.mutator_model)
        loggers['main'].info("Initialized target and mutator models")
        
        # 初始化评估器
        evaluator = RoBERTaEvaluator()
        loggers['main'].info("Initialized RoBERTa evaluator")
        
        # 初始化变异器
        mutator = LLMMutator(
            llm=mutator_model,
            temperature=args.temperature
        )
        loggers['main'].info("Initialized LLM mutator")
        
        # 初始化种子选择器
        seed_selector = UCBSeedSelector(
            seed_pool=initial_seeds,
            exploration_weight=args.exploration_weight
        )
        loggers['main'].info("Initialized UCB seed selector")
        
        # 初始化fuzzing引擎
        engine = FuzzingEngine(
            target_model=target_model,
            seed_selector=seed_selector,
            mutator=mutator,
            evaluator=evaluator,
            questions=harmful_questions,
            max_iterations=args.max_iterations,
            save_results=True,
            results_file=output_path / 'results.txt',
            success_file=output_path / 'successful_jailbreaks.csv',
            summary_file=output_path / 'experiment_summary.txt',
            loggers=loggers
        )
        loggers['main'].info("Initialized fuzzing engine")
        
        # 确保引擎运行前记录状态
        loggers['api'].info("Starting fuzzing engine...")
        
        # 运行fuzzing过程
        engine.run()
        
    except Exception as e:
        loggers['api'].error(f"Error in main: {str(e)}")
        raise

def load_seeds(seed_path):
    """加载种子数据"""
    try:
        df = pd.read_csv(seed_path)
        return df['text'].tolist()
    except Exception as e:
        raise RuntimeError(f"Failed to load seeds from {seed_path}: {e}")

if __name__ == '__main__':
    main()
