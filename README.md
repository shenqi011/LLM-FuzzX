# LLM-FuzzX

LLM-FuzzX 是一款面向大型语言模型（如 GPT、Claude、LLaMA）的用户友好型模糊测试工具，具备高级任务感知变异策略、精细化评估以及越狱检测功能，能够帮助研究人员和开发者快速发现潜在安全漏洞，并增强模型的稳健性。

## 主要特性

- 🚀 **用户友好的界面**: 提供直观的 Web 界面，支持可视化配置和实时监控
- 🔄 **多样化变异策略**: 支持多种高级变异方法，包括相似变异、交叉变异、扩展变异等
- 📊 **实时评估反馈**: 集成 RoBERTa 模型进行实时越狱检测和评估
- 🌐 **多模型支持**: 支持主流大语言模型，包括 GPT、Claude、LLaMA 等
- 📈 **可视化分析**: 提供种子流程图、实验数据统计等多维度分析功能
- 🔍 **细粒度日志**: 支持多级日志记录，包括主要日志、变异日志、越狱日志等

## 快速开始

### 环境要求

- Python 3.8+
- Node.js 14+
- Vue.js 3.x

### 后端安装

```bash
# 克隆项目
git clone https://github.com/Windy3f3f3f3f/LLM-FuzzX.git

# 安装依赖
cd LLM-FuzzX
pip install -r requirements.txt
```

### 前端安装

```bash
# 进入前端目录
cd llm-fuzzer-frontend

# 安装依赖
npm install

# 启动开发服务器
npm run serve
```

### 配置

1. 在 `config.py` 中配置相关 API 密钥：
```python
OPENAI_API_KEY = "your-openai-key"
CLAUDE_API_KEY = "your-claude-key"
```

2. 配置代理设置（如需要）：
```python
BASE_URL = "your-proxy-url"
```

## 使用指南

### 1. 启动服务

```bash
# 启动后端服务
python app.py

# 启动前端服务（新终端）
cd llm-fuzzer-frontend
npm run serve
```

### 2. 使用 Web 界面

1. 访问 `http://localhost:10001` 打开 Web 界面
2. 选择目标模型（如 GPT-3.5、Claude 等）
3. 配置测试参数：
   - 选择问题输入方式（默认问题集或自定义输入）
   - 设置最大迭代次数
   - 选择变异策略
   - 配置其他参数
4. 点击"运行测试"开始测试
5. 实时监控测试进度和结果

### 3. 查看结果

- 实验总结：包含整体测试统计和评估结果
- 详细结果：每次测试的具体响应和评估数据
- 成功详情：成功越狱案例的详细信息
- 种子流图：可视化展示种子变异过程

## 项目结构

```
LLM-FuzzX/
├── src/                    # 后端源代码
│   ├── api/               # API 接口
│   ├── evaluation/        # 评估模块
│   ├── fuzzing/          # 模糊测试核心
│   ├── models/           # 模型封装
│   └── utils/            # 工具函数
├── llm-fuzzer-frontend/   # 前端代码
├── scripts/               # 辅助脚本
├── data/                  # 数据文件
└── logs/                  # 日志文件
```

## 贡献指南

欢迎提交 Issue 和 Pull Request 来帮助改进项目。在提交 PR 前，请确保：

1. 代码符合项目的编码规范
2. 添加了必要的测试用例
3. 更新了相关文档

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 联系方式

如有问题或建议，欢迎通过以下方式联系：

- Issue: [GitHub Issues](https://github.com/Windy3f3f3f3f/LLM-FuzzX/issues)
- Email: wdwdwd1024@gmail.com
