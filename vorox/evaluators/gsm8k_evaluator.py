import re
import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset

from vorox.configs import EvaluatorType, EvaluatorConfig
from .evaluator_base import EvaluatorBase

class GSM8KEvaluator(EvaluatorBase):
    def __init__(self, config: EvaluatorConfig):
        super().__init__(config)
        self.repo_id = "openai/gsm8k"
        self.name = EvaluatorType.gsm8k
        self.performance_breakdown = {
            "all": {"num_correct": 0, "num_total": 0}
        }

        # Streaming test split
        self.dataset = load_dataset(self.repo_id, "main", split="test", streaming=True)

        # Grab first 3 rows for few-shot
        shots_iter = iter(load_dataset(self.repo_id, "main", split="test", streaming=True).take(3))
        self.few_shot_examples = list(shots_iter)

        def create_prompt(example):
            # Build few-shot prefix
            prompt_prefix = ""
            for shot in self.few_shot_examples:
                prompt_prefix += (
                    f"Q: {shot['question']}\n"
                    f"A: {shot['answer']}\n\n"
                )
            prompt_prefix += f"Q: {example['question']}\nA: "
            example["prompt"] = prompt_prefix
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
                # Run inference using model's __call__ function
                prompt = sample["prompt"]
                model_output = model(prompt)
                
                # Extract the final answer from model output
                # Look for the pattern "#### <number>" at the end
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
