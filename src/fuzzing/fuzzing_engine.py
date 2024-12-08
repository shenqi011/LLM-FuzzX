# src/fuzzing/fuzzing_engine.py
"""
FuzzingEngine class that orchestrates the fuzzing process.
"""

import logging
import json
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime
from src.utils.logger import (
    setup_logging,
    log_mutation,
    log_evaluation,
    log_success,
    log_stats
)
import csv
from pathlib import Path

@dataclass
class FuzzingResult:
    """数据类,用于存储每次fuzzing的结果"""
    iteration: int
    prompt: str
    response: str
    is_successful: bool
    timestamp: datetime
    metadata: Dict[str, Any] = None
    
    def to_csv_row(self) -> Dict[str, Any]:
        """转换为CSV行格式"""
        return {
            'iteration': self.iteration,
            'question': self.metadata.get('current_question', ''),  # 添加问题字段
            'prompt': self.prompt,
            'response': self.response, 
            'is_successful': self.is_successful,
            'timestamp': self.timestamp.isoformat(),
            'metadata': json.dumps(self.metadata or {}, ensure_ascii=False)
        }

class FuzzingEngine:
    """
    编排整个fuzzing过程的引擎类。
    协调seed selector、mutator和evaluator的工作。
    """

    def __init__(self, 
                 target_model,
                 seed_selector, 
                 mutator, 
                 evaluator,
                 questions: List[str],
                 max_iterations: int = 1000,
                 save_results: bool = True,
                 results_file: str = None,
                 success_file: str = None,
                 summary_file: str = None,
                 loggers: dict = None,
                 **kwargs):
        """
        初始化FuzzingEngine。

        Args:
            target_model: 目标LLM模型
            seed_selector: 种子选择策略
            mutator: 变异策略
            evaluator: 响应评估器
            questions: 问题列表
            max_iterations: 最大迭代次数
            save_results: 是否保存结果到文件
            results_file: 结果文件路径
            success_file: 成功案例文件路径
            summary_file: 实验总结文件路径
            loggers: 日志记录器
            **kwargs: 额外参数
        """
        self.loggers = loggers or {
            'main': logging.getLogger(__name__)
        }
        
        self.target_model = target_model
        self.seed_selector = seed_selector
        self.mutator = mutator
        self.evaluator = evaluator
        self.max_iterations = max_iterations
        self.save_results = save_results
        self.questions = questions
        
        # 存储所有fuzzing结果
        self.results: List[FuzzingResult] = []
        
        # 统计信息
        self.stats = {
            'total_attempts': 0,
            'successful_attempts': 0,
            'failed_attempts': 0,
            'mutation_types': {},  # 记录不同变异类型的使用次数
            'per_question_stats': {  # 新增:每个问题的统计
                question: {
                    'attempts': 0,
                    'successes': 0,
                    'success_rate': 0.0
                } for question in questions
            }
        }

        # 新增CSV文件相关
        self.success_file = success_file
        if self.success_file:
            # 确保目录存在
            Path(self.success_file).parent.mkdir(parents=True, exist_ok=True)
            # 创建CSV文件并写入表头
            with open(self.success_file, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'iteration', 'question', 'prompt', 'response', 
                    'is_successful', 'timestamp', 'metadata'
                ])
                writer.writeheader()

        # 添加JSON结果存储
        self.results_by_question = {
            question: [] for question in questions
        }

        self.summary_file = summary_file
        if self.summary_file:
            Path(self.summary_file).parent.mkdir(parents=True, exist_ok=True)

    def _save_success_to_csv(self, result: FuzzingResult):
        """保存成功案例到CSV"""
        if not self.success_file:
            return
            
        # 对每个成功的问题都写入一行
        with open(self.success_file, 'a', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'iteration', 'question', 'prompt', 'response',
                'is_successful', 'timestamp', 'metadata'
            ])
            
            for question, eval_result, response in zip(
                result.metadata['questions'],
                result.metadata['eval_results'],
                result.response
            ):
                if eval_result.get('is_successful', False):
                    row_data = {
                        'iteration': result.iteration,
                        'question': question,
                        'prompt': result.prompt,
                        'response': response,
                        'is_successful': True,
                        'timestamp': result.timestamp.isoformat(),
                        'metadata': json.dumps(eval_result, ensure_ascii=False)
                    }
                    writer.writerow(row_data)

    def _save_summary(self):
        """保存实验总结到txt文件"""
        if not self.summary_file:
            return
        
        with open(self.summary_file, 'w', encoding='utf-8') as f:
            # 写入总体统计
            f.write("=== Overall Statistics ===\n")
            f.write(f"Total Attempts: {self.stats['total_attempts']}\n")
            f.write(f"Total Successful Attempts: {self.stats['successful_attempts']}\n")
            overall_rate = self.get_success_rate()
            f.write(f"Overall Success Rate: {overall_rate:.2%}\n\n")
            
            # 写入每个问题的统计
            f.write("=== Per Question Statistics ===\n")
            for question, stats in self.stats['per_question_stats'].items():
                f.write(f"\nQuestion: {question}\n")
                f.write(f"Attempts: {stats['attempts']}\n")
                f.write(f"Successes: {stats['successes']}\n")
                f.write(f"Success Rate: {stats['success_rate']:.2%}\n")
                
            # 写入问题成功率排名
            f.write("\n=== Question Success Rate Ranking ===\n")
            sorted_questions = sorted(
                self.stats['per_question_stats'].items(),
                key=lambda x: x[1]['success_rate'],
                reverse=True
            )
            for i, (question, stats) in enumerate(sorted_questions, 1):
                f.write(f"\n{i}. Success Rate: {stats['success_rate']:.2%}\n")
                f.write(f"   Question: {question}\n")

    def run(self):
        """运行fuzzing过程"""
        self.loggers['main'].info('Starting fuzzing process...')
        
        try:
            for iteration in range(self.max_iterations):
                # 选择种子prompt
                seed = self.seed_selector.select_seed()
                
                # 变异种子prompt
                try:
                    mutated_prompts = self.mutator.mutate(seed)
                    if not mutated_prompts:
                        continue
                        
                    mutated_prompt = mutated_prompts[0]
                    
                    # 对每个问题进行测试
                    responses = []
                    eval_results = []
                    
                    for question in self.questions:
                        # 合成最终的prompt
                        final_prompt = mutated_prompt.replace('[INSERT PROMPT HERE]', question)
                        try:
                            response = self.target_model.generate(final_prompt)
                            responses.append(response)
                            # 评估响应
                            eval_result = self.evaluator.evaluate(response)
                            eval_results.append(eval_result)
                        except Exception as e:
                            self.loggers['main'].error(f'Error for question "{question}": {e}')
                            continue

                    # 如果所有问题都测试完成
                    if len(responses) == len(self.questions):
                        # 创建结果对象
                        result = FuzzingResult(
                            iteration=iteration + 1,
                            prompt=mutated_prompt,
                            response=responses,  # 存储所有响应
                            is_successful=any(r.get('is_successful', False) for r in eval_results),  # 任一成功即算成功
                            timestamp=datetime.now(),
                            metadata={
                                'eval_results': eval_results,
                                'questions': self.questions
                            }
                        )
                        
                        # 更新统计信息
                        self._update_stats(result)
                        self.results.append(result)
                        
                        # 更新种子选择器
                        self.seed_selector.update_stats(seed, result.is_successful)
                        
                        # 记录成功案例
                        if result.is_successful:
                            self._log_success(result)
                            self._save_success_to_csv(result)

                except Exception as e:
                    self.loggers['main'].error(f'Error in iteration {iteration}: {e}')
                    continue
                
        except KeyboardInterrupt:
            self.loggers['main'].info('Fuzzing process interrupted by user.')
        finally:
            self._log_final_stats()
            self._save_results_json()
            self._save_summary()

    def _update_stats(self, result: FuzzingResult):
        """更新统计信息"""
        self.stats['total_attempts'] += 1
        
        # 获取每个问题的评估结果
        eval_results = result.metadata['eval_results']
        questions = result.metadata['questions']
        
        # 更新每个问题的统计
        any_success = False
        for question, eval_result in zip(questions, eval_results):
            question_stats = self.stats['per_question_stats'][question]
            question_stats['attempts'] += 1
            
            if eval_result.get('is_successful', False):
                question_stats['successes'] += 1
                any_success = True
                
            # 更新成功率
            question_stats['success_rate'] = (
                question_stats['successes'] / question_stats['attempts']
            )
        
        # 更新总体统计
        if any_success:
            self.stats['successful_attempts'] += 1
        else:
            self.stats['failed_attempts'] += 1

        # 更新按问题分类的结果
        for question, eval_result, response in zip(
            result.metadata['questions'],
            result.metadata['eval_results'],
            result.response
        ):
            self.results_by_question[question].append({
                'iteration': result.iteration,
                'prompt': result.prompt,
                'response': response,
                'is_successful': eval_result.get('is_successful', False),
                'timestamp': result.timestamp.isoformat(),
                'metadata': eval_result
            })

    def _log_final_stats(self):
        """记录最终的统计信息"""
        final_stats = {
            'overall': {
                'total_attempts': self.stats['total_attempts'],
                'successful_attempts': self.stats['successful_attempts'],
                'failed_attempts': self.stats['failed_attempts'],
                'success_rate': self.get_success_rate(),
            },
            'mutation_distribution': self.stats['mutation_types'],
            'per_question_stats': {}
        }
        
        # 添加每个问题的详细统计
        for question, stats in self.stats['per_question_stats'].items():
            # 截断问题文本以便于显示
            question_short = question[:50] + '...' if len(question) > 50 else question
            final_stats['per_question_stats'][question_short] = {
                'attempts': stats['attempts'],
                'successes': stats['successes'],
                'success_rate': f"{stats['success_rate']:.2%}"
            }
        
        # 记录统计信息
        self.loggers['stats'].info("Final Statistics:")
        self.loggers['stats'].info(json.dumps(final_stats, indent=2, ensure_ascii=False))
        
        # 输出每个问题的成功率排名
        sorted_questions = sorted(
            self.stats['per_question_stats'].items(),
            key=lambda x: x[1]['success_rate'],
            reverse=True
        )
        
        self.loggers['stats'].info("\nQuestion Success Rates Ranking:")
        for i, (question, stats) in enumerate(sorted_questions, 1):
            self.loggers['stats'].info(
                f"{i}. Success Rate: {stats['success_rate']:.2%} "
                f"({stats['successes']}/{stats['attempts']}) "
                f"Question: {question[:100]}..."
            )

    def get_success_rate(self) -> float:
        """计算成功率"""
        if self.stats['total_attempts'] == 0:
            return 0.0
        return self.stats['successful_attempts'] / self.stats['total_attempts']

    def _log_success(self, result: FuzzingResult):
        """记录成功案例的详细信息"""
        successful_questions = []
        
        # 找出哪些问题成功了
        for question, eval_result in zip(
            result.metadata['questions'], 
            result.metadata['eval_results']
        ):
            if eval_result.get('is_successful', False):
                successful_questions.append(question)
        
        # 记录成功信息
        self.loggers['main'].info(
            f"SUCCESS\n"
            f"Iteration: {result.iteration}\n"
            f"Prompt Template: {result.prompt}\n"
            f"Successful Questions: {json.dumps(successful_questions, indent=2)}\n"
            f"Responses: {json.dumps(result.response, indent=2)}\n"
            f"Evaluation Results: {json.dumps(result.metadata['eval_results'], indent=2)}"
        )

    def _save_results_json(self):
        """保存所有结果到JSON文件"""
        if not self.success_file:
            return
        
        # 使用success_file的路径来确定JSON文件位置
        json_file = Path(self.success_file).parent / 'all_results.json'
        
        # 保存结果
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(
                self.results_by_question,
                f,
                ensure_ascii=False,
                indent=2
            )
        
        self.loggers['main'].info(f"Saved all results to {json_file}")
