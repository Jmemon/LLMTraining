"""
Module: dataset.py
Purpose: Implements a PyTorch IterableDataset that streams data from remote sources.

The dataset accepts:
- A list of remote addresses (URLs) for the dataset files.
- An optional transformation to apply to each raw sample.
- A prefetch buffer size.
- An optional metadata cache (now using PostgreSQL) for recording sample processing.
- An option to shuffle the buffer (to improve randomness while streaming).

Added logging to track the buffer size and timing of data pulls.
"""

import random
import time
from typing import List, Optional, Callable, Iterator
import logging
import torch
from torch.utils.data import IterableDataset

from data.remote_stream import open_remote
from data.metadata_cache import MetadataCache

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # For demonstration, set to DEBUG level.

class RemoteIterableDataset(IterableDataset):
    """
    A PyTorch IterableDataset that streams data from remote sources.

    This dataset supports reading from multiple remote file locations (s3, gcp, huggingface, etc.) specified as URLs.
    It supports basic prefetching and optional shuffling of a prefetch buffer.

    Parameters:
        addresses (List[str]): List of URLs pointing to the data files.
        transform (Optional[Callable]): Optional function to transform each raw sample.
        prefetch_size (int): Number of samples held in a prefetch buffer before yielding.
        cache (Optional[MetadataCache]): Instance of MetadataCache to record sample metadata per epoch.
        shuffle_buffer (bool): If True, the prefetch buffer will be shuffled before yielding.
        epoch (int): The current epoch number (used for metadata logging).
    """
    def __init__(
        self,
        addresses: List[str],
        transform: Optional[Callable] = None,
        prefetch_size: int = 1000,
        cache: Optional[MetadataCache] = None,
        shuffle_buffer: bool = False,
        epoch: int = 0
    ):
        self.addresses = addresses
        self.transform = transform
        self.prefetch_size = prefetch_size
        self.cache = cache
        self.shuffle_buffer = shuffle_buffer
        self.epoch = epoch

    def __iter__(self) -> Iterator:
        """
        Iterate over samples from the remote files with optional prefetching and shuffling.
        """
        buffer = []
        sample_counter = 0
        
        for address in self.addresses:
            logger.info(f"Opening remote file: {address}")
            try:
                start_file_time = time.monotonic()
                stream = open_remote(address)
            except Exception as e:
                logger.error(f"Failed to open remote file {address}: {e}")
                raise ValueError(f"Failed to open remote file {address}. Error: {e}")
            for line in stream:
                # Here we assume each line is a raw sample.
                sample = line.strip()
                if self.transform:
                    sample = self.transform(sample)
                buffer.append(sample)
                
                # Update metadata cache with current sample index.
                if self.cache:
                    self.cache.add_sample(self.epoch, sample_counter)
                sample_counter += 1

                logger.debug(f"Buffer size: {len(buffer)}")
                # When the buffer is full, yield items.
                if len(buffer) >= self.prefetch_size:
                    flush_time = time.monotonic()
                    duration = flush_time - start_file_time
                    logger.info(f"Flushing buffer with {len(buffer)} items after {duration:.2f} seconds")
                    if self.shuffle_buffer:
                        random.shuffle(buffer)
                    for item in buffer:
                        yield item
                    buffer.clear()
                    start_file_time = time.monotonic()  # Reset timer for next batch
        
            # Flush any remaining samples
            if buffer:
                flush_time = time.monotonic()
                duration = flush_time - start_file_time
                logger.info(f"End of file reached; flushing remaining {len(buffer)} items after {duration:.2f} seconds")
                if self.shuffle_buffer:
                    random.shuffle(buffer)
                for item in buffer:
                    yield item
                buffer.clear()

        logger.info(f"Iteration finished, total samples processed: {sample_counter}")
        
        # Ensure metadata cache is saved.
        if self.cache:
            self.cache.close() 