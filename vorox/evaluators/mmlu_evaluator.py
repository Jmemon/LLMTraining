import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset

from vorox.configs import EvaluatorType, EvaluatorConfig
from .evaluator_base import EvaluatorBase

class MMLUEvaluator(EvaluatorBase):
    def __init__(self, config: EvaluatorConfig):
        super().__init__(config)
        self.repo_id = "cais/mmlu"
        self.name = EvaluatorType.mmlu
        self.performance_breakdown = {}  # {subject: {"num_correct": int, "num_total": int}}
        
        # Load entire test set in streaming mode
        self.dataset = load_dataset(self.repo_id, "all", split="test", streaming=True)
        
        # Collect first 3 rows as shots for the few-shot prompt
        shots_iter = iter(load_dataset(self.repo_id, "all", split="test", streaming=True).take(3))
        self.few_shot_examples = list(shots_iter)

        # Apply prompt construction with dataset.map
        def create_prompt(example):
            # Build a few-shot prompt using the 3 stored shots
            prompt_prefix = ""
            for shot in self.few_shot_examples:
                # shot["question"], shot["choices"], shot["answer"], shot["subject"]
                correct_choice_idx = shot["answer"]
                correct_letter = chr(65 + correct_choice_idx)  # 0->A, 1->B, 2->C, 3->D
                
                # Format choices as A, B, C, D options
                choices_text = ""
                for i, choice in enumerate(shot["choices"]):
                    letter = chr(65 + i)  # A, B, C, D
                    choices_text += f"{letter}. {choice} "
                
                prompt_prefix += (
                    f"Q: {shot['question']}\n"
                    f"Choices: {choices_text}\n"
                    f"A: {correct_letter}\n\n"
                )
            
            # Format choices as A, B, C, D options for the current example
            choices_text = ""
            for i, choice in enumerate(example["choices"]):
                letter = chr(65 + i)  # A, B, C, D
                choices_text += f"{letter}. {choice} "
                
            prompt_prefix += f"Q: {example['question']}\nChoices: {choices_text}\nA: "
            example["prompt"] = prompt_prefix
            example["correct_idx"] = example["answer"]
            example["correct_letter"] = chr(65 + example["answer"])  # Store correct letter (A, B, C, D)
            return example

        self.dataset = self.dataset.map(create_prompt)

        # Build DataLoader (no shuffle for evaluation)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers
        )

    def __call__(self, model: nn.Module) -> dict:
        for batch in self.dataloader:
            # Suppose each batch is a list of dicts in streaming mode
            for sample in batch:
                subject = sample["subject"]
                if subject not in self.performance_breakdown:
                    self.performance_breakdown[subject] = {"num_correct": 0, "num_total": 0}

                # 1) Model inference:
                #    model_output = model.generate(sample["prompt"])  # however you run inference

                # 2) Convert model output to predicted choice
                #    pred_idx = figure_out_predicted_choice(...) 

                # 3) Compare pred_idx with sample["correct_idx"]
                #    if pred_idx == sample["correct_idx"]:
                #        self.performance_breakdown[subject]["num_correct"] += 1
                self.performance_breakdown[subject]["num_total"] += 1
        return self.performance_breakdown
