from langdetect import detect
import requests
import logging
from typing import Tuple
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from config import BING_TRANSLATE_API_KEY, BING_TRANSLATE_LOCATION

logger = logging.getLogger('main')

def _get_language_code(lang_code: str) -> str:
    """将语言代码转换为Bing翻译API支持的格式"""
    # 语言代码映射表
    LANG_MAP = {
        'zh': 'zh-Hans',  # 简体中文
        'en': 'en',       # 英语
        'ja': 'ja',       # 日语
        'ko': 'ko',       # 韩语
        'fr': 'fr',       # 法语
        'de': 'de',       # 德语
        'es': 'es',       # 西班牙语
        'ru': 'ru',       # 俄语
    }
    return LANG_MAP.get(lang_code, lang_code)

def translate_text(target: str, text: str) -> dict:
    """使用Bing翻译文本的函数
    
    Args:
        target: 目标语言代码 (例如: 'en', 'zh', 'ja')
        text: 要翻译的文本
        
    Returns:
        dict: 包含翻译结果的字典
        {
            "input": 原文,
            "translatedText": 译文,
            "detectedSourceLanguage": 检测到的源语言
        }
    """
    if not text.strip():
        raise ValueError("翻译文本不能为空")
        
    endpoint = "https://api.cognitive.microsofttranslator.com/translate"
    
    # 检测源语言
    try:
        source_lang = detect(text)
        source_lang = _get_language_code(source_lang)
    except Exception as e:
        logger.warning(f"语言检测失败: {str(e)}，将使用自动检测")
        source_lang = None
    
    # 转换目标语言代码
    target = _get_language_code(target)
    
    # 构建API请求参数
    params = {
        'api-version': '3.0',
        'to': target
    }
    # 如果成功检测到源语言，则添加到参数中
    if source_lang:
        params['from'] = source_lang
    
    # 构建请求头
    headers = {
        'Ocp-Apim-Subscription-Key': BING_TRANSLATE_API_KEY,
        'Ocp-Apim-Subscription-Region': BING_TRANSLATE_LOCATION,
        'Content-type': 'application/json'
    }
    
    # 构建请求体
    body = [{'text': text}]
    
    try:
        # 发送请求
        response = requests.post(endpoint, params=params, headers=headers, json=body)
        response.raise_for_status()  # 如果请求失败则抛出异常
        
        # 解析响应
        translations = response.json()
        
        # 构建返回结果
        result = {
            "input": text,
            "translatedText": translations[0]["translations"][0]["text"],
            "detectedSourceLanguage": translations[0].get("detectedLanguage", {}).get("language", source_lang or "auto")
        }
        
        logger.debug(f"Translation successful: {text} -> {result['translatedText']}")
        return result
        
    except requests.exceptions.RequestException as e:
        logger.error(f"翻译请求失败: {str(e)}")
        if hasattr(e.response, 'text'):
            logger.error(f"错误详情: {e.response.text}")
        raise
    except Exception as e:
        logger.error(f"翻译过程中发生未知错误: {str(e)}")
        raise

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type(Exception),
    retry_error_callback=lambda retry_state: fallback_return_original(retry_state.args[0])
)
def detect_and_translate(text: str) -> Tuple[str, str]:
    """检测文本语言并在需要时翻译成英文
    
    Args:
        text: 要翻译的文本
        
    Returns:
        Tuple[str, str]: (原文, 译文)
    """
    try:
        # 检测语言
        lang = detect(text)
        
        # 如果是英文,直接返回
        if lang == 'en':
            return text, text

        # 如果是其他语言,翻译成英文
        translation = translate_text('en', text)
        return text, translation['translatedText']
        
    except Exception as e:
        logger.error(f"翻译失败: {str(e)}")
        raise

def fallback_return_original(text: str) -> Tuple[str, str]:
    """重试失败后的回调函数"""
    logger.warning("所有重试都失败了，返回原文")
    return text, text
