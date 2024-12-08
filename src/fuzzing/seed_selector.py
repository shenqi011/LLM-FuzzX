# src/fuzzing/seed_selector.py
"""
种子选择策略的实现。
包含多种选择算法,用于从种子池中选择下一个待测试的种子。
"""

import random
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional, Dict
import logging

class SeedSelector(ABC):
    """
    种子选择器的基类,定义了选择策略的基本接口。
    """
    def __init__(self, seed_pool: Optional[List[str]] = None, **kwargs):
        """
        初始化种子选择器。

        Args:
            seed_pool: 初始种子池,如果为None则从文件加载
            **kwargs: 额外的参数
        """
        if seed_pool is not None:
            # 验证种子有效性
            self.seed_pool = [
                seed for seed in seed_pool 
                if isinstance(seed, str) and len(seed.strip()) > 0
            ]
            if not self.seed_pool:
                raise ValueError("种子池为空或所有种子无效")
            
            # 输出种子池信息
            logging.info(f"Loaded {len(self.seed_pool)} seeds")
            for i, seed in enumerate(self.seed_pool):
                logging.debug(f"Seed {i} (length={len(seed)}): {seed[:100]}...")
        else:
            # 从文件加载种子
            try:
                with open('data/seeds/initial_seeds.txt', 'r', encoding='utf-8') as f:
                    self.seed_pool = [line.strip() for line in f if line.strip()]
                if not self.seed_pool:
                    raise ValueError("种子文件为空")
            except FileNotFoundError:
                logging.error("种子文件未找到: data/seeds/initial_seeds.txt")
                raise
            
        # 初始化种子状态记录
        self.seed_stats = {
            seed: {
                'visits': 0,
                'successes': 0,
                'total_trials': 0
            } for seed in self.seed_pool
        }

    @abstractmethod
    def select_seed(self) -> str:
        """
        选择下一个种子。

        Returns:
            str: 选中的种子
        """
        pass

    def update_stats(self, seed: str, success: bool):
        """
        更新种子的统计信息。

        Args:
            seed: 使用的种子
            success: 是否成功
        """
        if seed in self.seed_stats:
            self.seed_stats[seed]['visits'] += 1
            self.seed_stats[seed]['total_trials'] += 1
            if success:
                self.seed_stats[seed]['successes'] += 1

    def get_success_rate(self, seed: str) -> float:
        """
        获取种子的成功率。

        Args:
            seed: 目标种子

        Returns:
            float: 成功率
        """
        stats = self.seed_stats[seed]
        if stats['total_trials'] == 0:
            return 0.0
        return stats['successes'] / stats['total_trials']

class RandomSeedSelector(SeedSelector):
    """
    随机选择策略。
    最单的选择方法,从种子池中随机选择一个种子。
    """
    def select_seed(self) -> str:
        if not self.seed_pool:
            raise ValueError("种子池为空")
        return random.choice(self.seed_pool)

class RoundRobinSeedSelector(SeedSelector):
    """
    轮询选择策略。
    按顺序循环选择种子,确保每个种子都有均等的机会被测试。
    """
    def __init__(self, seed_pool: Optional[List[str]] = None, **kwargs):
        super().__init__(seed_pool, **kwargs)
        self.current_index = 0

    def select_seed(self) -> str:
        if not self.seed_pool:
            raise ValueError("种子池为空")
        
        seed = self.seed_pool[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.seed_pool)
        return seed

class WeightedSeedSelector(SeedSelector):
    """
    基于权重的选择策略。
    根据种子的历史成功率分配选择权重。
    """
    def __init__(self, seed_pool: Optional[List[str]] = None, temperature: float = 1.0, **kwargs):
        """
        Args:
            seed_pool: 种子池
            temperature: 温度参数,控制探索与利用的平衡
            **kwargs: 额外参数
        """
        super().__init__(seed_pool, **kwargs)
        self.temperature = temperature

    def select_seed(self) -> str:
        if not self.seed_pool:
            raise ValueError("种子池为空")

        # 计算每个种子的权重
        weights = []
        for seed in self.seed_pool:
            success_rate = self.get_success_rate(seed)
            # 使用softmax计算权重
            weight = np.exp(success_rate / self.temperature)
            weights.append(weight)

        # 归一化权重
        weights = np.array(weights)
        weights = weights / np.sum(weights)

        # 按权重随机选择
        return np.random.choice(self.seed_pool, p=weights)

class UCBSeedSelector(SeedSelector):
    """
    基于UCB(Upper Confidence Bound)算法的选择策略。
    在探索和利用之间取得平衡。
    """
    def __init__(self, seed_pool: Optional[List[str]] = None, exploration_weight: float = 1.0, **kwargs):
        """
        Args:
            seed_pool: 种子池
            exploration_weight: 探索权重,控制探索项的重要性
            **kwargs: 额外参数
        """
        super().__init__(seed_pool, **kwargs)
        self.exploration_weight = exploration_weight
        self.total_trials = 0

    def select_seed(self) -> str:
        if not self.seed_pool:
            raise ValueError("种子池为空")

        self.total_trials += 1
        
        # 计算每个种子的UCB分数
        ucb_scores = []
        for seed in self.seed_pool:
            stats = self.seed_stats[seed]
            
            # 如果种子从未被访问过,给予最高分数
            if stats['visits'] == 0:
                ucb_scores.append(float('inf'))
                continue
            
            # UCB公式: 平均收益 + 探索项
            exploitation = self.get_success_rate(seed)
            exploration = np.sqrt(2 * np.log(self.total_trials) / stats['visits'])
            ucb_score = exploitation + self.exploration_weight * exploration
            ucb_scores.append(ucb_score)

        # 选择UCB分数最高的种子
        best_index = np.argmax(ucb_scores)
        return self.seed_pool[best_index]
