import argparse

import torch
import torch.nn as nn

from vorox.train import fit
from vorox.dummy_dataset import DummyDataset
from vorox.dummy_model import DummyModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    args = parser.parse_args()

    model = DummyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.MSELoss()

    train_loader = DummyDataset(100)
    val_loader = DummyDataset(100)

    fit(model, train_loader, val_loader, optimizer, loss_fn, args.epochs)
