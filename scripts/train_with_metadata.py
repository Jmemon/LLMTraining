"""
This script demonstrates how to integrate a PostgreSQL-backed metadata cache with a WebDataset data pipeline.
Each sample processed by the WebDataset pipeline is recorded via a mapping function that writes metadata (e.g. a sample index)
to the database. Note that if you are using multiple DataLoader workers, each worker should initialize its own connection.
"""

import os
import yaml
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import logging
import webdataset as wds

from vorox.config import Config
from vorox.loss import LossBase
from vorox.optimizer import OptimizerBase
from vorox.train import fit
from vorox.vorox import Vorox
from vorox.data.metadata_cache import MetadataCache

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

def create_record_metadata_mapper(cache_dsn: str, epoch: int):
    """
    Creates a mapper function that records metadata for each sample and then returns the sample unmodified.
    This mapper maintains a local counter in a mutable dictionary.
    
    Parameters:
      cache_dsn (str): PostgreSQL DSN used to open a metadata cache connection.
      epoch (int): Current epoch number.
    
    Returns:
      A function that takes a sample and returns it after recording metadata.
    """
    # Each worker calls this function to open its own PostgreSQL connection.
    cache = MetadataCache(cache_dsn)
    state = {"idx": 0}  # local counter encapsulated in a mutable dictionary

    def record_metadata(sample):
        # Record metadata for this sample
        idx = state["idx"]
        try:
            cache.add_sample(epoch, idx)
        except Exception as e:
            logger.error(f"Failed to record metadata for sample {idx}: {e}")
        state["idx"] += 1
        return sample

    return record_metadata

def main():
    # Load configuration from YAML using pydantic.
    with open("configs/20M_test_model.yml", "r") as f:
        cfg = Config.model_validate(yaml.safe_load(f))
    
    # Initialize the tokenizer from a pretrained model as specified in the config.
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer.name)
    
    # Build the model, optimizer, and loss function.
    model = Vorox(cfg, tokenizer.vocab_size)
    optimizer = OptimizerBase.build(model, cfg)
    loss_fn = LossBase.build(cfg)
    
    # Create the metadata mapper function using the configured PostgreSQL DSN.
    # Note: In a multi-worker setting, this function will be called per worker.
    record_metadata_fn = create_record_metadata_mapper(cfg.data.settings.cache_dsn, epoch=1)
    
    # Set up the dataset using WebDataset.
    # The data pipeline decodes raw bytes, extracts the "txt" field, maps the metadata (recording each sample),
    # shuffles with a provided buffer size, and groups samples into batches.
    dataset = (
        wds.WebDataset(cfg.data.source.urls)
        .shuffle(cfg.data.settings.prefetch_size)  # shuffle buffer size acts as prefetching window
        .decode("utf-8")                           # decode raw bytes to strings
        .to_tuple("txt")                           # assuming each sample has a "txt" key
        .map(record_metadata_fn)                   # record metadata for each sample
        .batched(cfg.train.batch_size)             # batch the samples
    )
    
    # Wrap the dataset in a DataLoader.
    # Note: since the dataset outputs batches already, we set batch_size=None.
    train_loader = DataLoader(
        dataset,
        num_workers=cfg.train.num_workers if hasattr(cfg.train, "num_workers") else 4,
        batch_size=None,
    )
    
    # Validation loader can be set up similarly if needed.
    val_loader = None  
    
    # Start training.
    fit(cfg, tokenizer, model, train_loader, val_loader, optimizer, loss_fn, cfg.train.epochs)

if __name__ == "__main__":
    main() 