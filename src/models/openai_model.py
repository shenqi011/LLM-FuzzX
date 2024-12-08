# src/models/openai_model.py
from src.models.llm_wrapper import LLMWrapper
import openai
from config import OPENAI_API_KEY, BASE_URL
from openai import OpenAI
import time
from functools import wraps
from typing import Callable, Any
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
import json
from src.utils.logger import setup_logging

def retry_with_exponential_backoff(
    max_retries: int = 5,
    initial_delay: float = 10,
    exponential_base: float = 2,
    max_delay: float = 60
) -> Callable:
    """
    用于API调用的指数退避重试装饰器
    
    Args:
        max_retries: 最大重试次数
        initial_delay: 初始延迟时间(秒)
        exponential_base: 指数基数
        max_delay: 最大延迟时间(秒)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            retry_count = 0
            delay = initial_delay

            while retry_count <= max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retry_count += 1
                    if retry_count > max_retries:
                        raise e

                    # 计算下一次重试的延迟时间
                    delay = min(delay * exponential_base, max_delay)
                    print(f"API调用失败，{delay}秒后进行第{retry_count}次重试...")
                    print(f"错误信息: {str(e)}")
                    time.sleep(delay)

        return wrapper
    return decorator

class OpenAIModel(LLMWrapper):
    """
    OpenAI API模型的包装类
    """
    def __init__(self, model_name: str, device: str = None, precision: str = None):
        """
        初始化OpenAI模型
        
        Args:
            model_name: OpenAI模型名称
            device: 不使用
            precision: 不使用
        """
        self.validate_model_name("openai", model_name)
        self.model_name = model_name
        self.gen_kwargs = {"model": model_name}
        
        # 使用传入的logger或创建新的logger
        self.logger = logging.getLogger('api')  # 改用api logger
        
        try:
            self.client = OpenAI(
                api_key=OPENAI_API_KEY,
                base_url=BASE_URL
            )
        except Exception as e:
            self.logger.error(f"初始化OpenAI客户端失败: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60)
    )
    def _make_api_call(self, is_completion: bool, **kwargs) -> str:
        """
        进行API调用的内部方法
        
        Args:
            is_completion: 是否是completion API
            **kwargs: API调用参数
            
        Returns:
            str: 生成的文本
        """
        try:
            if is_completion:
                response = self.client.completions.create(**kwargs)
                return response.choices[0].text
            else:
                chat_completion = self.client.chat.completions.create(**kwargs)
                return chat_completion.choices[0].message.content
        except Exception as e:
            logging.error(f"OpenAI API调用失败: {str(e)}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60)
    )
    def generate(self, prompt: str, max_new_tokens: int = None, temperature: float = None, **kwargs) -> str:
        """
        生成文本
        
        Args:
            prompt: 输入提示
            max_new_tokens: 最大生成token数
            temperature: 温度参数
            **kwargs: 其他参数
            
        Returns:
            str: 生成的文本
        """
        api_logger = logging.getLogger('api')
        
        try:
            # 确保prompt不为None且是字符串
            if prompt is None:
                raise ValueError("Prompt cannot be None")
                
            if isinstance(prompt, list):
                prompt = prompt[0] if prompt else ""
                
            if not isinstance(prompt, str):
                prompt = str(prompt)
                
            # 记录API调用开始
            api_logger.info(json.dumps({
                'event': 'api_call_start',
                'model': self.model_name,
                'prompt_length': len(prompt)
            }))
            
            if max_new_tokens:
                self.gen_kwargs["max_tokens"] = max_new_tokens
            if temperature:
                self.gen_kwargs["temperature"] = temperature
            
            # 更新其他参数
            self.gen_kwargs.update(kwargs)
            
            # 根据模型类型选择不同的API调用方式
            if self.model_name == "gpt-3.5-turbo-instruct":
                api_kwargs = {
                    "prompt": prompt,
                    **self.gen_kwargs
                }
                response = self._make_api_call(is_completion=True, **api_kwargs)
            else:
                messages = [{"role": "user", "content": prompt}]  # content必须是字符串
                api_kwargs = {
                    "messages": messages,
                    **self.gen_kwargs
                }
                response = self._make_api_call(is_completion=False, **api_kwargs)
            
            # 记录API调用成功
            api_logger.info(json.dumps({
                'event': 'api_call_success',
                'model': self.model_name,
                'prompt_length': len(prompt),
                'response_length': len(response)
            }))
            
            return response
            
        except Exception as e:
            # 记录API调用失败
            api_logger.error(json.dumps({
                'event': 'api_call_error',
                'model': self.model_name,
                'error': str(e),
                'prompt': prompt[:100] if prompt else None  # 记录prompt信息以便调试
            }))
            raise
