import torch
import torch.nn as nn


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, size):
        self.size = size

    def __iter__(self):
        for i in range(self.size):
            yield torch.randn(10)

    def __len__(self):
        return self.size


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
