import pytest
from src.fuzzing.mutator import BaseMutator, NLPMutator, LLMMutator, HybridMutator, MutationStrategy
from src.models.llm_wrapper import LLMWrapper, retry_with_exponential_backoff
import logging

class MockLLM(LLMWrapper):
    """用于测试的模拟LLM"""
    def generate(self, prompt: str, **kwargs) -> str:
        # 根据不同的提示词返回不同的模拟响应
        if "similar" in prompt.lower():
            return "This is a similar template with [INSERT PROMPT HERE] marker"
        elif "crossover" in prompt.lower():
            return "This is a crossover template with [INSERT PROMPT HERE] marker"
        elif "add sentences" in prompt.lower():
            return "Additional context sentences."
        elif "condense" in prompt.lower():
            return "Shortened template with [INSERT PROMPT HERE] marker"
        elif "rephrase" in prompt.lower():
            return "Rephrased template with [INSERT PROMPT HERE] marker"
        return "Default response with [INSERT PROMPT HERE]"

@pytest.fixture
def mock_llm():
    return MockLLM()

@pytest.fixture
def llm_mutator(mock_llm):
    return LLMMutator(llm=mock_llm)

@pytest.fixture
def nlp_mutator():
    return NLPMutator()

@pytest.fixture
def hybrid_mutator(nlp_mutator, llm_mutator):
    return HybridMutator(nlp_mutator=nlp_mutator, llm_mutator=llm_mutator)

class TestLLMMutator:
    """测试LLM变异器"""
    
    def test_validate_prompt(self, llm_mutator):
        """测试prompt验证"""
        # 有效的prompt
        valid_prompt = "Test prompt with [INSERT PROMPT HERE] marker"
        assert llm_mutator._validate_prompt(valid_prompt) == True
        
        # 无效的prompt - 没有placeholder
        invalid_prompt = "Test prompt without marker"
        assert llm_mutator._validate_prompt(invalid_prompt) == False
        
        # 无效的prompt - 词数太少
        short_prompt = "[INSERT PROMPT HERE]"
        assert llm_mutator._validate_prompt(short_prompt) == False

    def test_mutate_similar(self, llm_mutator):
        """测试相似变异"""
        prompt = "Original prompt with [INSERT PROMPT HERE] marker"
        mutations = llm_mutator.mutate_similar(prompt)
        
        assert len(mutations) > 0
        for mutation in mutations:
            assert "[INSERT PROMPT HERE]" in mutation
            assert mutation != prompt

    def test_mutate_crossover(self, llm_mutator):
        """测试交叉变异"""
        prompt1 = "First prompt with [INSERT PROMPT HERE] marker"
        prompt2 = "Second prompt with [INSERT PROMPT HERE] marker"
        mutations = llm_mutator.mutate_crossover(prompt1, prompt2)
        
        assert len(mutations) > 0
        for mutation in mutations:
            assert "[INSERT PROMPT HERE]" in mutation
            assert mutation != prompt1
            assert mutation != prompt2

    def test_mutate_expand(self, llm_mutator):
        """测试扩展变异"""
        prompt = "Original prompt with [INSERT PROMPT HERE] marker"
        mutations = llm_mutator.mutate_expand(prompt)
        
        assert len(mutations) > 0
        for mutation in mutations:
            assert "[INSERT PROMPT HERE]" in mutation
            assert len(mutation) > len(prompt)

    def test_mutate_shorten(self, llm_mutator):
        """测试缩短变异"""
        prompt = "This is a very long and verbose prompt that needs to be shortened with [INSERT PROMPT HERE] marker"
        mutations = llm_mutator.mutate_shorten(prompt)
        
        assert len(mutations) > 0
        for mutation in mutations:
            assert "[INSERT PROMPT HERE]" in mutation

    def test_mutate_rephrase(self, llm_mutator):
        """测试重述变异"""
        prompt = "Original prompt with [INSERT PROMPT HERE] marker"
        mutations = llm_mutator.mutate_rephrase(prompt)
        
        assert len(mutations) > 0
        for mutation in mutations:
            assert "[INSERT PROMPT HERE]" in mutation
            assert mutation != prompt

    def test_mutate_error_handling(self, llm_mutator, caplog):
        """测试错误处理"""
        # 模拟LLM失败的情况
        class FailingLLM(LLMWrapper):
            def __init__(self):
                self.call_count = 0
                
            @retry_with_exponential_backoff(
                max_retries=3,
                initial_delay=0.1  # 测试时使用较短的延迟
            )
            def generate(self, prompt: str, **kwargs):
                self.call_count += 1
                raise Exception("API error")  # 始终失败
        
        failing_mutator = LLMMutator(llm=FailingLLM())
        prompt = "Test prompt with [INSERT PROMPT HERE] marker"
        
        with caplog.at_level(logging.ERROR):
            mutations = failing_mutator.mutate_similar(prompt)  # 测试单个变异方法
            assert len(mutations) == 1  # 应该返回原始prompt
            assert mutations[0] == prompt  # 验证返回原始prompt
            assert "LLM similar mutation" in caplog.text  # 验证错误日志
            assert "API error" in caplog.text  # 验证错误消息

    def test_llm_mutator_with_retry(self):
        """测试LLM变异器的重试机制"""
        class FailingLLM(LLMWrapper):
            def __init__(self):
                self.call_count = 0
                
            @retry_with_exponential_backoff(
                max_retries=3,
                initial_delay=0.1  # 测试时使用较短的延迟
            )
            def generate(self, prompt: str, **kwargs):
                self.call_count += 1
                if self.call_count < 3:  # 前两次调用失败
                    raise Exception("API error")
                return "Success response with [INSERT PROMPT HERE]"
                
        llm = FailingLLM()
        mutator = LLMMutator(llm=llm)
        
        prompt = "Test prompt with [INSERT PROMPT HERE]"
        mutations = mutator.mutate_similar(prompt)
        
        assert len(mutations) > 0
        assert llm.call_count >= 3  # 验证重试机制生效

class TestHybridMutator:
    """测试混合变异器"""
    
    def test_nlp_only_strategy(self, hybrid_mutator):
        """测试仅NLP策略"""
        hybrid_mutator.strategy = MutationStrategy.NLP_ONLY
        prompt = "Test prompt with [INSERT PROMPT HERE] marker"
        mutations = hybrid_mutator.mutate(prompt)
        
        assert len(mutations) > 0
        for mutation in mutations:
            assert "[INSERT PROMPT HERE]" in mutation

    def test_llm_only_strategy(self, hybrid_mutator):
        """测试仅LLM策略"""
        hybrid_mutator.strategy = MutationStrategy.LLM_ONLY
        prompt = "Test prompt with [INSERT PROMPT HERE] marker"
        mutations = hybrid_mutator.mutate(prompt)
        
        assert len(mutations) > 0
        for mutation in mutations:
            assert "[INSERT PROMPT HERE]" in mutation

    def test_hybrid_strategy(self, hybrid_mutator):
        """测试混合策略"""
        hybrid_mutator.strategy = MutationStrategy.HYBRID
        prompt = "Test prompt with [INSERT PROMPT HERE] marker"
        mutations = hybrid_mutator.mutate(prompt)
        
        assert len(mutations) > 0
        for mutation in mutations:
            assert "[INSERT PROMPT HERE]" in mutation

    def test_invalid_prompt(self, hybrid_mutator):
        """测试无效prompt"""
        invalid_prompt = "Invalid prompt without marker"
        mutations = hybrid_mutator.mutate(invalid_prompt)
        
        assert len(mutations) == 1
        assert mutations[0] == invalid_prompt

if __name__ == "__main__":
    pytest.main(["-v"]) 
