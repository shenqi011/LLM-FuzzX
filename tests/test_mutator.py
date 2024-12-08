import pytest
from src.fuzzing.mutator import (
    Mutator,
    SynonymReplacementMutator,
    CharacterReplacementMutator,
    TextShufflingMutator,
    CompositeMutator
)

# 测试用的种子prompt
@pytest.fixture
def seed_prompt():
    return "Let's have a discussion about [INSERT PROMPT HERE] in a creative way"

@pytest.fixture
def invalid_prompt():
    return "Invalid prompt without placeholder"

@pytest.fixture
def short_prompt():
    return "[INSERT PROMPT HERE]"

class TestMutatorBase:
    """测试Mutator基类的功能"""
    
    def test_validate_prompt(self, seed_prompt, invalid_prompt, short_prompt):
        mutator = SynonymReplacementMutator()  # 使用一个具体的子类来测试
        
        # 测试有效的prompt
        assert mutator._validate_prompt(seed_prompt) is True
        
        # 测试无占位符的prompt
        assert mutator._validate_prompt(invalid_prompt) is False
        
        # 测试过短的prompt
        # 修改测试用例，因为我们现在使用 min_words 来控制最小词数
        assert mutator._validate_prompt(short_prompt) is False, \
            "A prompt containing only the placeholder should be invalid"

class TestSynonymReplacementMutator:
    """测试同义词替换变异器"""
    
    def test_initialization(self):
        mutator = SynonymReplacementMutator(mutation_rate=0.5)
        assert mutator.mutation_rate == 0.5
        assert mutator.placeholder == '[INSERT PROMPT HERE]'
    
    def test_get_synonyms(self):
        mutator = SynonymReplacementMutator()
        synonyms = mutator._get_synonyms('happy')
        assert isinstance(synonyms, list)
        assert len(synonyms) > 0
        assert 'happy' not in synonyms  # 同义词列表不应包含原词
    
    def test_mutate_preserves_placeholder(self, seed_prompt):
        mutator = SynonymReplacementMutator(mutation_rate=1.0)  # 100%变异率
        mutated = mutator.mutate(seed_prompt)
        assert '[INSERT PROMPT HERE]' in mutated
        assert mutated != seed_prompt  # 应该有所变化
    
    def test_mutate_with_zero_rate(self, seed_prompt):
        mutator = SynonymReplacementMutator(mutation_rate=0.0)  # 0%变异率
        mutated = mutator.mutate(seed_prompt)
        assert mutated == seed_prompt  # 不应该有变化

class TestCharacterReplacementMutator:
    """测试字符替换变异器"""
    
    def test_initialization(self):
        mutator = CharacterReplacementMutator(mutation_rate=0.5)
        assert mutator.mutation_rate == 0.5
        assert 'a' in mutator.char_map
    
    def test_mutate_preserves_placeholder(self, seed_prompt):
        mutator = CharacterReplacementMutator(mutation_rate=1.0)
        mutated = mutator.mutate(seed_prompt)
        assert '[INSERT PROMPT HERE]' in mutated
    
    def test_character_replacement(self):
        mutator = CharacterReplacementMutator(mutation_rate=1.0)
        test_text = "test"
        mutated = mutator.mutate(test_text + " [INSERT PROMPT HERE]")
        assert mutated != test_text  # 应该有所变化
        assert '[INSERT PROMPT HERE]' in mutated

class TestTextShufflingMutator:
    """测试文本重排变异器"""
    
    def test_initialization(self):
        mutator = TextShufflingMutator(preserve_first_last=False)
        assert mutator.preserve_first_last is False
    
    def test_mutate_preserves_placeholder(self, seed_prompt):
        mutator = TextShufflingMutator()
        mutated = mutator.mutate(seed_prompt)
        assert '[INSERT PROMPT HERE]' in mutated
    
    def test_preserve_first_last(self, seed_prompt):
        mutator = TextShufflingMutator(preserve_first_last=True)
        original_words = seed_prompt.split()
        mutated_words = mutator.mutate(seed_prompt).split()
        
        # 检查首尾词是否保持不变
        assert mutated_words[0] == original_words[0]
        assert mutated_words[-1] == original_words[-1]
    
    def test_short_prompt_unchanged(self, short_prompt):
        mutator = TextShufflingMutator()
        mutated = mutator.mutate(short_prompt)
        assert mutated == short_prompt

class TestCompositeMutator:
    """测试复合变异器"""
    
    @pytest.fixture
    def composite_mutator(self):
        mutators = [
            SynonymReplacementMutator(),
            CharacterReplacementMutator(),
            TextShufflingMutator()
        ]
        return CompositeMutator(mutators=mutators)
    
    def test_initialization(self, composite_mutator):
        assert len(composite_mutator.mutators) == 3
        assert isinstance(composite_mutator.mutators[0], SynonymReplacementMutator)
    
    def test_sequential_mutation(self, seed_prompt):
        mutators = [
            CharacterReplacementMutator(mutation_rate=1.0),
            TextShufflingMutator()
        ]
        composite = CompositeMutator(mutators=mutators, sequential=True)
        mutated = composite.mutate(seed_prompt)
        assert mutated != seed_prompt
        assert '[INSERT PROMPT HERE]' in mutated
    
    def test_random_mutation(self, seed_prompt, composite_mutator):
        # 测试多次,确保随机选择正常工作
        results = set()
        for _ in range(10):
            mutated = composite_mutator.mutate(seed_prompt)
            results.add(mutated)
            assert '[INSERT PROMPT HERE]' in mutated
        
        # 应该得到不同的结果(虽然理论上可能相同)
        assert len(results) > 1

def test_integration(seed_prompt):
    """集成测试:测试所有变异器的组合使用"""
    mutators = [
        SynonymReplacementMutator(mutation_rate=0.3),
        CharacterReplacementMutator(mutation_rate=0.3),
        TextShufflingMutator()
    ]
    
    # 测试顺序应用
    sequential = CompositeMutator(mutators=mutators, sequential=True)
    seq_result = sequential.mutate(seed_prompt)
    assert '[INSERT PROMPT HERE]' in seq_result
    
    # 测试随机选择
    random = CompositeMutator(mutators=mutators, sequential=False)
    rand_result = random.mutate(seed_prompt)
    assert '[INSERT PROMPT HERE]' in rand_result 

