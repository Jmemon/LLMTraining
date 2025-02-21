from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from vorox.config import Config


def fit(
    cfg: Config,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
):
    model = model.to(cfg.device)
    for ep in tqdm(range(cfg.train.epochs)):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            batch = batch.to(cfg.device)
            batch_split = torch.chunk(batch, cfg.train.micro_batch_size, dim=0)
            for micro_batch in batch_split:
                outputs = model(micro_batch)
                loss = loss_fn(outputs, micro_batch)
                train_loss += loss.item()
                loss.backward()
            optimizer.step()
            exit()

        model.eval()
        with torch.no_grad():
            for batch in tqdm(val_loader):
                outputs = model(batch)
                loss = loss_fn(outputs, batch)
                val_loss += loss.item()

        train_loss, val_loss = train_loss / len(train_loader), val_loss / len(val_loader)
        print(f"Epoch {ep+1}, Train Loss: {train_loss}, Validation Loss: {val_loss}")

    return model
