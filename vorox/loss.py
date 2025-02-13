
import torch.nn as nn

from vorox.config import LossType


class LossBase:
    @classmethod
    def build(cls, cfg):
        if cfg.loss.type == LossType.mse:
            return nn.MSELoss()
        elif cfg.loss.type == LossType.cross_entropy or cfg.loss.type == LossType.perplexity:
            return nn.CrossEntropyLoss()
