import pytest
import logging

@pytest.fixture(autouse=True)
def setup_logging():
    """设置日志配置"""
    logging.basicConfig(
        level=logging.ERROR,
        format='%(levelname)s: %(message)s'
    )
