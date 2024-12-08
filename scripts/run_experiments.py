# scripts/run_experiments.py
"""
Script to run experiments.
"""

from main import main
import sys

# 不同的实验配置
configs = [
    ["--target_model", "gpt-3.5-turbo", "--max_iterations", "50"],
    ["--target_model", "gpt-4", "--max_iterations", "50"],
]

for config in configs:
    sys.argv[1:] = config  # 设置命令行参数
    main()  # 运行实验
