import re
import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset

from vorox.configs import EvaluatorType, EvaluatorConfig
from .evaluator_base import EvaluatorBase

class GSMSymbolicEvaluator(EvaluatorBase):
    def __init__(self, config: EvaluatorConfig):
        super().__init__(config)
        self.repo_id = "apple/GSM-Symbolic"
        self.name = EvaluatorType.gsm_symbolic
        self.performance_breakdown = {"all": {"num_correct": 0, "num_total": 0}}

        # subset="p1", split="test", streaming mode
        self.dataset = load_dataset(self.repo_id, "p1", split="test", streaming=True)

        # Grab first 3 rows for few-shot
        shots_iter = iter(load_dataset(self.repo_id, "p1", split="test", streaming=True).take(3))
        few_shots = list(shots_iter)

        def create_prompt(example):
            # Construct the multi-shot prompt
            prompt = "As an expert problem solver, solve step by step the following mathematical questions.\n"
            for i, shot in enumerate(few_shots, start=1):
                # shot["question"], shot["answer"] ends with "#### <answer>"
                # separate textual reasoning vs final answer from shot["answer"]
                prompt += (f"Q: {shot['question']}\n"
                           f"A: Let's think step by step. {shot['answer']}. ")
            prompt += f"\n\nQ: {example['question']}\nA: Let's think step by step."
            example["prompt"] = prompt
            example["ground_truth"] = example["answer"]
            return example

        self.dataset = self.dataset.map(create_prompt)

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers
        )

    def __call__(self, model: nn.Module) -> dict:
        for batch in self.dataloader:
            for sample in batch:
                # Run model inference using model's __call__ function
                prompt = sample["prompt"]
                model_output = model(prompt)
                
                # Extract the final answer from model output
                # Look for the pattern "#### <answer>" at the end
                predicted_answer = None
                match = re.search(r"####\s*([\d\.\-+]+)", model_output)
                if match:
                    predicted_answer = match.group(1).strip()
                
                # Extract ground truth answer
                ground_truth = sample["ground_truth"]
                correct_answer = None
                match = re.search(r"####\s*([\d\.\-+]+)", ground_truth)
                if match:
                    correct_answer = match.group(1).strip()
                
                # Compare answers
                if predicted_answer is not None and correct_answer is not None:
                    if predicted_answer == correct_answer:
                        self.performance_breakdown["all"]["num_correct"] += 1
                
                self.performance_breakdown["all"]["num_total"] += 1
                
        return self.performance_breakdown
