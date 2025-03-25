from typing import Dict
import torch
from torch import nn
from torch.utils.data import DataLoader
from vorox.configs import EvaluatorType, EvaluatorConfig  # or whichever config class is relevant

class EvaluatorBase:
    def __init__(self, config: EvaluatorConfig):
        """
        Initialize the evaluator.
        """
        self.repo_id = ""
        self.name = None
        self.performance_breakdown = {}
        self.dataloader = None
        self.config = config

    def __call__(self, model: nn.Module) -> Dict[str, Dict[str, int]]:
        """
        Iterate over the dataloader, constructing prompts, running model inference,
        and evaluating. Update and return performance_breakdown.
        """
        raise NotImplementedError()
