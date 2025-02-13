import yaml

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from vorox.config import Config
from vorox.dclm_baseline import BabyDCLMBaseline, collate_fn
from vorox.loss import LossBase
from vorox.optimizer import OptimizerBase
from vorox.train import fit
from vorox.vorox import Vorox

if __name__ == "__main__":
    with open("configs/20M_test_model.yml", "r") as f:
        cfg = Config.model_validate(yaml.safe_load(f))

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer.name)
    model = Vorox(cfg, tokenizer.vocab_size)
    optimizer = OptimizerBase.build(model, cfg)
    loss_fn = LossBase.build(cfg)

    val_loader = None
    train_loader = DataLoader(
        BabyDCLMBaseline(macro_batch_size=cfg.train.batch_size, num_batches=10), 
        batch_size=cfg.train.batch_size, 
        shuffle=True,
        collate_fn=collate_fn
    )

    fit(cfg, tokenizer, model, train_loader, val_loader, optimizer, loss_fn, cfg.train.epochs)
