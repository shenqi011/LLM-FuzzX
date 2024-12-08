"""
测试FuzzingEngine的功能
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime
import os
import tempfile

from src.fuzzing.fuzzing_engine import FuzzingEngine, FuzzingResult

@pytest.fixture
def mock_components():
    """提供模拟的组件"""
    return {
        'target_model': Mock(),
        'seed_selector': Mock(),
        'mutator': Mock(),
        'evaluator': Mock()
    }

@pytest.fixture
def engine(mock_components):
    """提供配置好的FuzzingEngine实例"""
    return FuzzingEngine(
        target_model=mock_components['target_model'],
        seed_selector=mock_components['seed_selector'],
        mutator=mock_components['mutator'],
        evaluator=mock_components['evaluator'],
        max_iterations=5
    )

@pytest.fixture
def temp_results_file():
    """提供临时结果文件"""
    with tempfile.NamedTemporaryFile(delete=False) as f:
        temp_file = f.name
    yield temp_file
    os.unlink(temp_file)  # 测试后清理

def test_engine_initialization(mock_components):
    """测试引擎初始化"""
    engine = FuzzingEngine(**mock_components, max_iterations=10)
    
    assert engine.target_model == mock_components['target_model']
    assert engine.seed_selector == mock_components['seed_selector']
    assert engine.mutator == mock_components['mutator']
    assert engine.evaluator == mock_components['evaluator']
    assert engine.max_iterations == 10
    assert isinstance(engine.results, list)
    assert len(engine.results) == 0
    assert engine.stats['total_attempts'] == 0
    assert engine.stats['successful_attempts'] == 0
    assert engine.stats['failed_attempts'] == 0

def test_fuzzing_process(engine, mock_components):
    """测试完整的fuzzing过程"""
    # 配置mock组件的行为
    mock_components['seed_selector'].select_seed.return_value = "test_seed"
    mock_components['mutator'].mutate.return_value = "mutated_prompt"
    mock_components['target_model'].generate.return_value = "model_response"
    mock_components['evaluator'].evaluate.return_value = {
        'is_successful': True,
        'metadata': {'reason': 'test'}
    }

    # 运行fuzzing过程
    engine.run()

    # 验证组件调用
    assert mock_components['seed_selector'].select_seed.call_count == 5
    assert mock_components['mutator'].mutate.call_count == 5
    assert mock_components['target_model'].generate.call_count == 5
    assert mock_components['evaluator'].evaluate.call_count == 5
    
    # 验证结果存储
    assert len(engine.results) == 5
    assert all(isinstance(r, FuzzingResult) for r in engine.results)

def test_error_handling(engine, mock_components):
    """测试错误处理"""
    # 模拟模型生成错误
    mock_components['target_model'].generate.side_effect = Exception("Model error")
    
    # 运行fuzzing过程
    engine.run()
    
    # 验证错误不会中断整个过程
    assert mock_components['seed_selector'].select_seed.call_count == 5
    assert len(engine.results) == 0  # 因为所有生成都失败了

def test_result_saving(temp_results_file, mock_components):
    """测试结果保存功能"""
    # 创建带有特定结果文件的引擎
    engine = FuzzingEngine(
        **mock_components,
        max_iterations=1,
        results_file=temp_results_file
    )
    
    # 配置mock返回值
    mock_components['seed_selector'].select_seed.return_value = "test_seed"
    mock_components['mutator'].mutate.return_value = "mutated_prompt"
    mock_components['target_model'].generate.return_value = "model_response"
    mock_components['evaluator'].evaluate.return_value = {
        'is_successful': True,
        'metadata': {'reason': 'test'}
    }
    
    # 运行fuzzing过程
    engine.run()
    
    # 验证结果文件内容
    with open(temp_results_file, 'r', encoding='utf-8') as f:
        content = f.read()
        assert 'Iteration: 1' in content
        assert 'Prompt: mutated_prompt' in content
        assert 'Response: model_response' in content
        assert 'Success: True' in content

def test_stats_tracking(engine, mock_components):
    """测试统计信息跟踪"""
    # 配置mock使一半的尝试成功
    mock_components['evaluator'].evaluate.side_effect = [
        {'is_successful': True},
        {'is_successful': False},
        {'is_successful': True},
        {'is_successful': False},
        {'is_successful': True}
    ]
    
    # 运行fuzzing过程
    engine.run()
    
    # 验证统计信息
    assert engine.stats['total_attempts'] == 5
    assert engine.stats['successful_attempts'] == 3
    assert engine.stats['failed_attempts'] == 2
    assert engine.get_success_rate() == 0.6

def test_keyboard_interrupt_handling(engine, mock_components):
    """测试键盘中断处理"""
    # 模拟键盘中断
    mock_components['seed_selector'].select_seed.side_effect = KeyboardInterrupt()
    
    # 运行fuzzing过程
    engine.run()
    
    # 验证正确处理了中断
    assert mock_components['seed_selector'].select_seed.call_count == 1
    assert len(engine.results) == 0

@pytest.mark.parametrize("success_count,expected_rate", [
    (0, 0.0),
    (3, 0.6),
    (5, 1.0)
])
def test_success_rate_calculation(engine, mock_components, success_count, expected_rate):
    """测试成功率计算"""
    # 配置mock返回值
    results = [{'is_successful': i < success_count} for i in range(5)]
    mock_components['evaluator'].evaluate.side_effect = results
    
    # 运行fuzzing过程
    engine.run()
    
    # 验证成功率计算
    assert engine.get_success_rate() == expected_rate

def test_fuzzing_result_dataclass():
    """测试FuzzingResult数据类"""
    result = FuzzingResult(
        iteration=1,
        prompt="test_prompt",
        response="test_response",
        is_successful=True,
        timestamp=datetime.now(),
        metadata={'test': 'data'}
    )
    
    assert result.iteration == 1
    assert result.prompt == "test_prompt"
    assert result.response == "test_response"
    assert result.is_successful is True
    assert isinstance(result.timestamp, datetime)
    assert result.metadata == {'test': 'data'} 
