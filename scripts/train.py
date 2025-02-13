import yaml

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from vorox.config import Config
from vorox.dclm_baseline import BabyDCLM
from vorox.train import fit
from vorox.vorox import Vorox

if __name__ == "__main__":
    with open("configs/test.yml", "r") as f:
        cfg = Config.model_validate(yaml.safe_load(f))

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer.name)
    model = Vorox(cfg, tokenizer.vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.learning_rate)
    loss_fn = nn.MSELoss()

    val_loader = None
    train_loader = DataLoader(BabyDCLM(), batch_size=cfg.train.batch_size, shuffle=True)

    fit(tokenizer, model, train_loader, val_loader, optimizer, loss_fn, cfg.train.epochs)
