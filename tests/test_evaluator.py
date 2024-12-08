"""
测试评估器功能
"""

import pytest
from src.evaluation.roberta_evaluator import RoBERTaEvaluator
from src.evaluation.evaluator import ResponseEvaluator

@pytest.fixture
def roberta_evaluator():
    """提供RoBERTa评估器实例"""
    return RoBERTaEvaluator()

@pytest.fixture
def response_evaluator():
    """提供基础响应评估器实例"""
    return ResponseEvaluator()

def test_roberta_evaluator_single(roberta_evaluator):
    """测试RoBERTa评估器的单个评估"""
    response = "I cannot help with harmful activities."
    result = roberta_evaluator.evaluate(response)
    
    assert isinstance(result, dict)
    assert "is_successful" in result
    assert "confidence" in result
    assert isinstance(result["is_successful"], bool)
    assert 0 <= result["confidence"] <= 1

def test_roberta_evaluator_batch(roberta_evaluator):
    """测试RoBERTa评估器的批量评估"""
    responses = [
        "I cannot help with that.",
        "Here's how you can accomplish that task..."
    ]
    results = roberta_evaluator.evaluate(responses)
    
    assert isinstance(results, list)
    assert len(results) == len(responses)
    assert all("is_successful" in r for r in results)

def test_response_evaluator(response_evaluator):
    """测试基础响应评估器"""
    # 测试应该被拒绝的响应
    reject_response = "I cannot help with that request."
    reject_result = response_evaluator.evaluate(reject_response)
    assert not reject_result["is_successful"]
    
    # 测试可能成功的响应
    success_response = "Here are the steps to follow..."
    success_result = response_evaluator.evaluate(success_response)
    assert success_result["is_successful"]

def test_evaluator_error_handling(roberta_evaluator):
    """测试错误处理"""
    with pytest.raises(Exception):
        roberta_evaluator.evaluate("")  # 空输入
        
    with pytest.raises(Exception):
        roberta_evaluator.evaluate(None)  # 无效输入

@pytest.mark.parametrize("response,expected", [
    ("I cannot help with that.", False),
    ("Here's how to do it...", True),
    ("I am unable to assist.", False),
    ("Follow these steps...", True),
])
def test_evaluator_responses(response_evaluator, response, expected):
    """测试不同类型的响应"""
    result = response_evaluator.evaluate(response)
    assert result["is_successful"] == expected 
