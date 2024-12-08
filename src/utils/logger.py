# src/utils/logger.py
"""
Logger setup and utilities.
"""

import logging
from pathlib import Path
from typing import Optional
import json
from datetime import datetime

def setup_logging(log_dir: str = "logs", log_name: str = None) -> logging.Logger:
    """
    设置统一的日志系统
    
    Args:
        log_dir: 日志目录
        log_name: 日志文件名前缀,如果不指定则使用时间戳
        
    Returns:
        logging.Logger: 配置好的logger
    """
    # 创建日志目录
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成日志文件名
    if not log_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_name = f"fuzzing_{timestamp}"
        
    # 创建logger
    logger = logging.getLogger("fuzzing")
    logger.setLevel(logging.DEBUG)
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 创建并配置文件处理器 - 完整日志
    full_handler = logging.FileHandler(
        log_dir / f"{log_name}_full.log",
        encoding='utf-8'
    )
    full_handler.setFormatter(formatter)
    full_handler.setLevel(logging.DEBUG)
    logger.addHandler(full_handler)
    
    # 创建并配置文件处理器 - 成功案例
    success_handler = logging.FileHandler(
        log_dir / f"{log_name}_success.log",
        encoding='utf-8'
    )
    success_handler.setFormatter(formatter)
    success_handler.setLevel(logging.INFO)
    success_handler.addFilter(lambda record: 'SUCCESS' in record.getMessage())
    logger.addHandler(success_handler)
    
    # 创建并配置文件处理器 - 评估结果
    eval_handler = logging.FileHandler(
        log_dir / f"{log_name}_eval.log",
        encoding='utf-8'
    )
    eval_handler.setFormatter(formatter)
    eval_handler.setLevel(logging.INFO)
    eval_handler.addFilter(lambda record: 'EVAL' in record.getMessage())
    logger.addHandler(eval_handler)
    
    # 创建并配置控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    
    return logger

def log_mutation(logger: logging.Logger, mutation_type: str, original: str, mutated: str):
    """记录变异操作"""
    logger.debug(
        f"MUTATION [{mutation_type}]\n"
        f"Original: {original[:100]}...\n"
        f"Mutated: {mutated[:100]}..."
    )

def log_evaluation(logger: logging.Logger, prompt: str, response: str, result: dict):
    """记录评估结果"""
    logger.info(
        f"EVAL\n"
        f"Prompt: {prompt[:100]}...\n"
        f"Response: {response[:100]}...\n"
        f"Result: {json.dumps(result, ensure_ascii=False)}"
    )

def log_success(logger: logging.Logger, prompt: str, response: str, metadata: dict = None):
    """记录成功案例"""
    logger.info(
        f"SUCCESS\n"
        f"Prompt: {prompt}\n"
        f"Response: {response}\n"
        f"Metadata: {json.dumps(metadata, ensure_ascii=False) if metadata else '{}'}"
    )

def log_stats(logger: logging.Logger, stats: dict):
    """记录统计信息"""
    logger.info(f"STATS {json.dumps(stats, ensure_ascii=False)}")

def get_logger(name: str) -> logging.Logger:
    """获取指定名称的logger"""
    return logging.getLogger(name)

# 添加日志级别常量
LOG_LEVELS = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}
