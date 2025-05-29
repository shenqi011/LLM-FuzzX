import unittest
from src.utils.language_utils import translate_text, detect_and_translate
import requests
from unittest.mock import patch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestTranslation(unittest.TestCase):
    def setUp(self):
        """测试开始前的设置"""
        self.test_cases = {
            "chinese": {
                "text": "你好，世界",
                "expected_lang": "zh",
                "target_lang": "en",
                "expected_translation": "Hello, world"
            },
            "english": {
                "text": "Hello, world",
                "expected_lang": "en",
                "target_lang": "zh",
                "expected_translation": "你好，世界"
            },
            "japanese": {
                "text": "こんにちは、世界",
                "expected_lang": "ja",
                "target_lang": "en",
                "expected_translation": "Hello, world"
            }
        }

    def test_translate_text_basic(self):
        """测试基本翻译功能"""
        logger.info("Testing basic translation functionality...")
        
        for lang, case in self.test_cases.items():
            logger.info(f"Testing translation for {lang}")
            try:
                result = translate_text(case["target_lang"], case["text"])
                
                self.assertIsInstance(result, dict)
                self.assertIn("translatedText", result)
                self.assertIn("input", result)
                
                logger.info(f"Input: {result['input']}")
                logger.info(f"Translation: {result['translatedText']}")
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Translation API request failed: {str(e)}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error during translation: {str(e)}")
                raise

    def test_detect_and_translate(self):
        """测试语言检测和翻译组合功能"""
        logger.info("Testing language detection and translation...")
        
        for lang, case in self.test_cases.items():
            logger.info(f"Testing detection and translation for {lang}")
            try:
                original, translated = detect_and_translate(case["text"])
                
                self.assertEqual(original, case["text"])
                self.assertIsInstance(translated, str)
                
                logger.info(f"Original: {original}")
                logger.info(f"Translated: {translated}")
                
            except Exception as e:
                logger.error(f"Error in detect_and_translate for {lang}: {str(e)}")
                raise

    def test_error_handling(self):
        """测试错误处理"""
        logger.info("Testing error handling...")
        
        # 测试空字符串
        with self.assertRaises(Exception):
            translate_text("en", "")
        
        # 测试无效的目标语言
        with self.assertRaises(Exception):
            translate_text("invalid_lang", "Hello")
        
        # 测试超长文本
        long_text = "test" * 5000
        try:
            result = translate_text("en", long_text)
            logger.info("Long text translation succeeded")
        except Exception as e:
            logger.warning(f"Long text translation failed as expected: {str(e)}")

    @patch('src.utils.language_utils.requests.post')
    def test_api_failure(self, mock_post):
        """测试API失败情况"""
        logger.info("Testing API failure scenarios...")
        
        # 模拟API超时
        mock_post.side_effect = requests.Timeout("Connection timed out")
        with self.assertRaises(requests.Timeout):
            translate_text("en", "Hello")
        
        # 模拟API返回错误
        mock_post.side_effect = requests.RequestException("API Error")
        with self.assertRaises(requests.RequestException):
            translate_text("en", "Hello")

if __name__ == '__main__':
    unittest.main() 