import pytest
from src.models.openai_model import OpenAIModel
from src.models.claude_model import ClaudeModel
from src.models.llama_model import LlamaModel

def test_llama_model():
    model = LlamaModel(model_name="meta-llama/Llama-2-7b-chat-hf")
    response = model.generate("Hello, how are you?")
    assert isinstance(response, str)
    assert len(response) > 0

def test_openai_model():
    model = OpenAIModel(model_name="gpt-3.5-turbo")
    response = model.generate("Hello, how are you?")
    assert isinstance(response, str)
    assert len(response) > 0

def test_claude_model():
    model = ClaudeModel(model_name="claude-3-sonnet-20240229")
    response = model.generate("Hello, how are you?")
    assert isinstance(response, str)
    assert len(response) > 0

def test_model_error_handling():
    with pytest.raises(ValueError):
        OpenAIModel(model_name="invalid-model")
    with pytest.raises(ValueError):
        ClaudeModel(model_name="invalid-model")
    with pytest.raises(ValueError):
        LlamaModel(model_name="invalid-model")

@pytest.mark.parametrize("model_class,model_name", [
    (OpenAIModel, "gpt-3.5-turbo"),
    (ClaudeModel, "claude-3-sonnet-20240229"),
    (LlamaModel, "meta-llama/Llama-2-7b-chat-hf")
])
def test_model_parameters(model_class, model_name):
    model = model_class(model_name=model_name)
    
    # 测试基本参数设置
    model.temperature = 0.7
    assert model.temperature == 0.7
    
    model.max_tokens = 100
    assert model.max_tokens == 100
    
    # 测试生成结果
    response = model.generate("Test message")
    assert isinstance(response, str)
    assert len(response) > 0

# 添加conftest.py中的标记定义
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "skip_if_no_openai: mark test to skip if OpenAI credentials are not available"
    )
    config.addinivalue_line(
        "markers", "skip_if_no_claude: mark test to skip if Claude credentials are not available"
    )
    config.addinivalue_line(
        "markers", "skip_if_no_huggingface: mark test to skip if Hugging Face model is not available"
    )
