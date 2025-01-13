# LLM-FuzzX

[中文文档](README_zh.md)

LLM-FuzzX is an open-source, user-friendly fuzzing tool for Large Language Models (like GPT, Claude, LLaMA) featuring advanced task-aware mutation strategies, fine-grained evaluation, and jailbreak detection capabilities. It helps researchers and developers quickly identify potential security vulnerabilities and enhance model robustness. The methodology is primarily based on [LLM-Fuzzer](https://www.usenix.org/conference/usenixsecurity24/presentation/yu-jiahao).

## Key Features

- 🚀 **User-Friendly Interface**: Intuitive web interface with visual configuration and real-time monitoring
- 🔄 **Diverse Mutation Strategies**: Support for various advanced mutation methods, including similar mutation, crossover mutation, expansion mutation, etc.
- 📊 **Real-time Evaluation Feedback**: Integrated RoBERTa model for real-time jailbreak detection and evaluation
- 🌐 **Multi-model Support**: Compatible with mainstream LLMs including GPT, Claude, LLaMA, etc.
- 📈 **Visualization Analysis**: Multi-dimensional analysis with seed flow diagrams and experimental data statistics
- 🔍 **Fine-grained Logging**: Support for multi-level logging, including main logs, mutation logs, jailbreak logs, etc.

## System Architecture

LLM-FuzzX adopts a front-end and back-end separated architecture design, consisting of the following core modules:

### Core Engine Layer
- **Fuzzing Engine**: System's central scheduler, coordinating component workflows
- **Seed Management**: Responsible for seed storage, retrieval, and updates
- **Model Interface**: Unified model calling interface supporting multiple model implementations
- **Evaluation System**: RoBERTa-based jailbreak detection and multi-dimensional evaluation

### Mutation Strategies
- **Similar Mutation**: Maintains original template style while generating similar structured variants
- **Crossover Mutation**: Combines templates selected from the seed pool
- **Expansion Mutation**: Adds supplementary content to original templates
- **Shortening Mutation**: Generates more concise variants through compression and refinement
- **Restatement Mutation**: Rephrases while maintaining semantic meaning
- **Target-aware Mutation**: Generates variants based on target model characteristics

## Quick Start

### Requirements

- Python 3.8+
- Node.js 14+
- CUDA support (for RoBERTa evaluation model)
- 8GB+ system memory
- Stable network connection

### Backend Installation

```bash
# Clone the project
git clone https://github.com/Windy3f3f3f3f/LLM-FuzzX.git

# Create virtual environment
conda create -n llm-fuzzx python=3.10
conda activate llm-fuzzx

# Install dependencies
cd LLM-FuzzX
pip install -r requirements.txt
```

### Frontend Installation

```bash
# Enter frontend directory
cd llm-fuzzer-frontend

# Install dependencies
npm install

# Start development server
npm run serve
```

### Configuration

1. Create `.env` file in project root to configure API keys:
```bash
OPENAI_API_KEY=your-openai-key
CLAUDE_API_KEY=your-claude-key
HUGGINGFACE_API_KEY=your-huggingface-key
```

2. Configure model parameters in `config.py`:
```python
MODEL_CONFIG = {
    'target_model': 'gpt-3.5-turbo',
    'mutator_model': 'gpt-3.5-turbo',
    'evaluator_model': 'roberta-base',
    'temperature': 0.7,
    'max_tokens': 2048
}
```

## Usage Guide

### 1. Start Services

```bash
# Start backend service
python app.py  # Default runs on http://localhost:10003

# Start frontend service
cd llm-fuzzer-frontend
npm run serve  # Default runs on http://localhost:10001
```

### 2. Basic Usage Flow

1. Select target test model (supports GPT, Claude, LLaMA, etc.)
2. Prepare test data
   - Use preset question sets
   - Custom input questions
3. Configure test parameters
   - Set maximum iteration count
   - Select mutation strategies
   - Configure evaluation thresholds
4. Start testing and monitor in real-time
   - View current progress
   - Monitor success rate
   - Analyze mutation effects

### 3. Result Analysis

The system provides multi-level logging:
- `main.log`: Main processes and key events
- `mutation.log`: Mutation operation records
- `jailbreak.log`: Successful jailbreak cases
- `error.log`: Errors and exceptions

## Project Structure

```
LLM-FuzzX/
├── src/                    # Backend source code
│   ├── api/               # API interfaces
│   ├── evaluation/        # Evaluation module
│   ├── fuzzing/          # Fuzzing core
│   ├── models/           # Model wrappers
│   └── utils/            # Utility functions
├── llm-fuzzer-frontend/   # Frontend code
├── scripts/               # Helper scripts
├── data/                  # Data files
└── logs/                  # Log files
```

## Best Practices

1. Test Scale Settings
   - Recommended to limit single test iterations to under 1000
   - Start with small-scale trials for new scenarios
   - Adjust concurrency based on available resources

2. Mutation Strategy Selection
   - Prefer single mutation strategy for simple scenarios
   - Combine multiple mutation methods for complex scenarios
   - Maintain balance in mutation intensity

3. Resource Optimization
   - Set reasonable API call intervals
   - Clean historical records periodically
   - Monitor system resource usage

## Contributing

Welcome to participate in the project through:
1. Submit Issues
   - Report bugs
   - Suggest new features
   - Share usage experiences
2. Submit Pull Requests
   - Fix issues
   - Add features
   - Improve documentation
3. Methodology Contributions
   - Provide new mutation strategies
   - Design innovative evaluation methods
   - Share testing experiences

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

- Issue: [GitHub Issues](https://github.com/Windy3f3f3f3f/LLM-FuzzX/issues)
- Email: wdwdwd1024@gmail.com

## References

[1] Yu, J., Lin, X., Yu, Z., & Xing, X. (2024). LLM-Fuzzer: Scaling Assessment of Large Language Model Jailbreaks. In 33rd USENIX Security Symposium (USENIX Security 24) (pp. 4657-4674). USENIX Association.
