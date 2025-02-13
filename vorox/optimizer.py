
import torch.optim as optim

from vorox.config import OptimizerType


class OptimizerBase:

    @classmethod
    def build(cls, model, cfg):
        if cfg.optimizer.type == OptimizerType.adamw:
            return optim.AdamW(model.parameters(), lr=cfg.optimizer.lr, betas=cfg.optimizer.betas, weight_decay=cfg.optimizer.weight_decay)
        elif cfg.optimizer.type == OptimizerType.adam:
            return optim.Adam(model.parameters(), lr=cfg.optimizer.lr, betas=cfg.optimizer.betas, weight_decay=cfg.optimizer.weight_decay)
        elif cfg.optimizer.type == OptimizerType.sgd:
            return optim.SGD(model.parameters(), lr=cfg.optimizer.lr)
