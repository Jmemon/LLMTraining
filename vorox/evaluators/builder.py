from typing import List
from vorox.configs import EvaluatorType, EvaluatorConfig
from .mmlu_evaluator import MMLUEvaluator
from .gsm8k_evaluator import GSM8KEvaluator
from .gsmsymbolic_evaluator import GSMSymbolicEvaluator

class EvaluatorBuilder:
    @staticmethod
    def build(config: EvaluatorConfig, eval_types: List[EvaluatorType]):
        evaluators = []
        for eval_type in eval_types:
            if eval_type == EvaluatorType.mmlu:
                evaluators.append(MMLUEvaluator(config))
            elif eval_type == EvaluatorType.gsm8k:
                evaluators.append(GSM8KEvaluator(config))
            elif eval_type == EvaluatorType.gsm_symbolic:
                evaluators.append(GSMSymbolicEvaluator(config))
            else:
                raise ValueError(f"Invalid evaluator type: {eval_type}")
        return evaluators
