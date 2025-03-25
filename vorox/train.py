from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from vorox.configs import RunConfig as Config


def fit(
    cfg: Config,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
):
    """
    Trains a neural language model using gradient-based optimization with gradient accumulation.
    
    Implements an epoch-based training loop with optional validation, supporting micro-batch
    gradient accumulation for memory-efficient training of large models.
    
    Architecture:
        - Implements standard supervised learning with O(epochs * dataset_size) time complexity
        - Uses gradient accumulation pattern with O(macro_batch_size / micro_batch_size) accumulation steps
        - Supports mixed-precision training when configured in the model
        - Memory complexity scales with O(micro_batch_size * max_seq_len) rather than macro_batch_size
    
    Interface:
        - cfg (Config): Complete configuration object containing training parameters,
          device settings, and batch sizing strategy. Must include train.epochs (int > 0),
          train.micro_batch_size (int > 0), and device (Device enum) fields.
        - model (nn.Module): PyTorch neural network model to be trained. Must implement
          forward() method compatible with the input data format from train_loader.
        - train_loader (DataLoader): PyTorch DataLoader providing training batches.
          Each batch must be compatible with model's forward() method and loss_fn.
          Determines number of optimization steps per epoch.
        - val_loader (DataLoader): Optional PyTorch DataLoader providing validation batches.
          Pass None to skip validation. When provided, must yield batches compatible
          with model's forward() method and loss_fn.
        - optimizer (optim.Optimizer): PyTorch optimizer initialized with model parameters.
          Controls gradient-based parameter updates according to the configured algorithm.
        - loss_fn (nn.Module): PyTorch loss function module compatible with model outputs
          and expected targets. Must be callable with signature loss_fn(outputs, targets).
    
    Behavior:
        - Stateless between calls; maintains no persistent state beyond function execution
        - Not thread-safe; designed for single-process execution
        - Manages model training state transitions (train/eval modes)
        - Accumulates gradients across micro-batches before applying optimizer step
        - Reports per-epoch metrics via console output
        - WARNING: Contains a debugging exit() call that terminates after first batch
    
    Integration:
        - Typically called from training scripts after model, data, and optimizer initialization
        - Example:
          ```
          model = Vorox(cfg, tokenizer.vocab_size)
          optimizer = OptimizerBase.build(model, cfg)
          loss_fn = LossBase.build(cfg)
          fit(cfg, model, train_loader, val_loader, optimizer, loss_fn)
          ```
    
    Limitations:
        - No checkpoint saving or early stopping mechanisms
        - No learning rate scheduling support
        - No distributed training capabilities
        - No progress metrics beyond epoch-level loss reporting
        - Contains an exit() call that prevents training beyond first batch (debugging artifact)
        - Validation loss calculation assumes val_loader is not None
    
    Returns:
        nn.Module: The trained model instance, moved to the configured device.
    """
    model = model.to(cfg.device)
    for ep in tqdm(range(cfg.train.epochs)):
        model.train()
        train_loss = 0
        val_loss = 0
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
