"""
测试种子选择器的各项功能
"""

import pytest
import numpy as np
from src.fuzzing.seed_selector import (
    SeedSelector,
    RandomSeedSelector,
    RoundRobinSeedSelector,
    WeightedSeedSelector,
    UCBSeedSelector
)

@pytest.fixture
def sample_seeds():
    """提供测试用的种子样本"""
    return ['seed1', 'seed2', 'seed3', 'seed4', 'seed5']

@pytest.fixture
def random_selector(sample_seeds):
    """提供RandomSeedSelector实例"""
    return RandomSeedSelector(seed_pool=sample_seeds)

@pytest.fixture
def round_robin_selector(sample_seeds):
    """提供RoundRobinSeedSelector实例"""
    return RoundRobinSeedSelector(seed_pool=sample_seeds)

@pytest.fixture
def weighted_selector(sample_seeds):
    """提供WeightedSeedSelector实例"""
    return WeightedSeedSelector(seed_pool=sample_seeds, temperature=1.0)

@pytest.fixture
def ucb_selector(sample_seeds):
    """提供UCBSeedSelector实例"""
    return UCBSeedSelector(seed_pool=sample_seeds, exploration_weight=1.0)

def test_seed_selector_initialization(sample_seeds):
    """测试种子选择器的���始化"""
    # 测试使用提供的种子池初始化
    selector = RandomSeedSelector(seed_pool=sample_seeds)
    assert selector.seed_pool == sample_seeds
    
    # 测试统计信息的初始化
    assert all(
        selector.seed_stats[seed] == {'visits': 0, 'successes': 0, 'total_trials': 0}
        for seed in sample_seeds
    )

def test_random_selector(random_selector, sample_seeds):
    """测试随机选择器"""
    # 测试多次选择都返回有效的种子
    for _ in range(10):
        seed = random_selector.select_seed()
        assert seed in sample_seeds

def test_round_robin_selector(round_robin_selector, sample_seeds):
    """测试轮询选择器"""
    # 测试是否按顺序返回所有种子
    selected_seeds = [round_robin_selector.select_seed() for _ in range(len(sample_seeds))]
    assert selected_seeds == sample_seeds
    
    # 测试是否循环返回
    next_seed = round_robin_selector.select_seed()
    assert next_seed == sample_seeds[0]

def test_weighted_selector(weighted_selector, sample_seeds):
    """测试加权选择器"""
    # 更新一些统计信息
    weighted_selector.update_stats(sample_seeds[0], True)
    weighted_selector.update_stats(sample_seeds[1], False)
    
    # 测试选择结果是否有效
    seed = weighted_selector.select_seed()
    assert seed in sample_seeds
    
    # 测试成功率计算
    assert weighted_selector.get_success_rate(sample_seeds[0]) == 1.0
    assert weighted_selector.get_success_rate(sample_seeds[1]) == 0.0

def test_ucb_selector(ucb_selector, sample_seeds):
    """测试UCB选择器"""
    # 第一次选择应该返回未访问过的种子
    first_seed = ucb_selector.select_seed()
    assert first_seed in sample_seeds
    
    # 更新一些统计信息
    ucb_selector.update_stats(first_seed, True)
    
    # 再次选择应该倾向于选择未访问过的种子
    second_seed = ucb_selector.select_seed()
    assert second_seed in sample_seeds
    assert second_seed != first_seed  # 应该选择一个不同的种子

def test_stats_update():
    """测试统计信息更新"""
    selector = RandomSeedSelector(seed_pool=['test_seed'])
    
    # 测试成功案例的更新
    selector.update_stats('test_seed', True)
    assert selector.seed_stats['test_seed']['visits'] == 1
    assert selector.seed_stats['test_seed']['successes'] == 1
    assert selector.seed_stats['test_seed']['total_trials'] == 1
    
    # 测试失败案例的更新
    selector.update_stats('test_seed', False)
    assert selector.seed_stats['test_seed']['visits'] == 2
    assert selector.seed_stats['test_seed']['successes'] == 1
    assert selector.seed_stats['test_seed']['total_trials'] == 2

def test_empty_seed_pool():
    """测试空种子池的处理"""
    selector = RandomSeedSelector(seed_pool=[])
    with pytest.raises(ValueError, match="种子池为空"):
        selector.select_seed()

def test_success_rate_calculation(random_selector, sample_seeds):
    """测试成功率计算"""
    seed = sample_seeds[0]
    
    # 测试初始成功率
    assert random_selector.get_success_rate(seed) == 0.0
    
    # 测试成功案例后的成功率
    random_selector.update_stats(seed, True)
    assert random_selector.get_success_rate(seed) == 1.0
    
    # 测试混合案例的成功率
    random_selector.update_stats(seed, False)
    assert random_selector.get_success_rate(seed) == 0.5

@pytest.mark.parametrize("selector_class", [
    RandomSeedSelector,
    RoundRobinSeedSelector,
    WeightedSeedSelector,
    UCBSeedSelector
])
def test_selector_interfaces(selector_class, sample_seeds):
    """测试所有选择器是否正确实现了接口"""
    selector = selector_class(seed_pool=sample_seeds)
    
    # 测试基本功能
    seed = selector.select_seed()
    assert seed in sample_seeds
    
    # 测试统计更新
    selector.update_stats(seed, True)
    assert selector.seed_stats[seed]['visits'] == 1
    assert selector.seed_stats[seed]['successes'] == 1
