import logging
from src.utils.language_utils import translate_text
import json
import requests
from config import BING_TRANSLATE_API_KEY, BING_TRANSLATE_LOCATION

# 设置详细的日志记录
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_api_configuration():
    """测试API配置是否正确"""
    logger.info("检查API配置...")
    if not BING_TRANSLATE_API_KEY or BING_TRANSLATE_API_KEY == "not_use":
        logger.error("BING_TRANSLATE_API_KEY 未设置或无效")
        return False
    if not BING_TRANSLATE_LOCATION:
        logger.error("BING_TRANSLATE_LOCATION 未设置")
        return False
    logger.info(f"API Key: {BING_TRANSLATE_API_KEY[:5]}...")
    logger.info(f"Location: {BING_TRANSLATE_LOCATION}")
    return True

def test_simple_translation():
    """测试简单的翻译功能"""
    test_text = "你好，世界"
    source_lang = "zh-Hans"  # 简体中文
    target_lang = "en"
    
    try:
        # 首先测试API配置
        if not test_api_configuration():
            return False

        logger.info(f"开始测试翻译: '{test_text}' ({source_lang} -> {target_lang})")
        
        # 构建请求参数
        endpoint = "https://api.cognitive.microsofttranslator.com/translate"
        params = {
            'api-version': '3.0',
            'from': source_lang,
            'to': target_lang
        }
        headers = {
            'Ocp-Apim-Subscription-Key': BING_TRANSLATE_API_KEY,
            'Ocp-Apim-Subscription-Region': BING_TRANSLATE_LOCATION,
            'Content-type': 'application/json'
        }
        body = [{'text': test_text}]

        # 记录请求信息
        logger.debug("Request details:")
        logger.debug(f"Endpoint: {endpoint}")
        logger.debug(f"Parameters: {params}")
        logger.debug(f"Headers: {headers}")
        logger.debug(f"Body: {json.dumps(body, ensure_ascii=False)}")
        
        # 直接使用requests发送请求
        response = requests.post(endpoint, params=params, headers=headers, json=body)
        
        # 记录响应信息
        logger.debug(f"Response status: {response.status_code}")
        logger.debug(f"Response headers: {dict(response.headers)}")
        logger.debug(f"Response content: {response.text}")
        
        # 检查响应状态
        response.raise_for_status()
        
        # 解析响应
        result = response.json()
        logger.info("Translation result:")
        logger.info(json.dumps(result, ensure_ascii=False, indent=2))
        
        return True
        
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP错误: {str(e)}")
        logger.error(f"响应内容: {e.response.text if hasattr(e, 'response') else 'No response content'}")
        return False
    except Exception as e:
        logger.error(f"翻译过程中发生错误: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    print("开始测试翻译API...")
    success = test_simple_translation()
    if success:
        print("测试成功完成！")
    else:
        print("测试失败！") 