import torch


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, size):
        self.size = size

    def __iter__(self):
        for i in range(self.size):
            yield torch.randn(10)

    def __len__(self):
        return self.size
