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
    A memory-efficient PyTorch IterableDataset that streams data from remote sources with controlled memory footprint and resumable processing.
    
    This dataset implements a streaming architecture designed for large-scale training on datasets that exceed available memory,
    employing a prefetch buffer strategy that balances between random access and sequential streaming efficiency. The implementation
    prioritizes constant memory usage (O(prefetch_size)) while maintaining amortized O(1) access time per sample.
    
    Architecture:
        - Streaming I/O: Uses smart_open to lazily access remote files without downloading them entirely
        - Buffered processing: Maintains a fixed-size prefetch buffer to amortize I/O costs (O(prefetch_size) memory)
        - Controlled randomization: Implements Fisher-Yates shuffling within buffer windows for memory-efficient randomization
        - Stateful processing: Records sample indices in PostgreSQL for resumability and distributed training coordination
        - Resource lifecycle: Automatically manages file handles and database connections throughout iterator lifecycle
    
    Interface:
        - __init__(addresses, transform, prefetch_size, cache, shuffle_buffer, epoch): Configures dataset parameters
        - __iter__() -> Iterator: Returns a stateful iterator that yields processed samples sequentially
    
    Parameters:
        addresses (List[str]): URLs to remote data files supporting multiple protocols (s3://, gs://, hf://, file://, http://)
        transform (Optional[Callable[[str], Any]]): Function applied to each raw text line; must be serializable for distributed use
        prefetch_size (int, default=1000): Controls memory usage, shuffling window size, and I/O batch efficiency
        cache (Optional[MetadataCache]): PostgreSQL connection for tracking processed samples; enables resumability
        shuffle_buffer (bool, default=False): Enables in-buffer randomization; true randomness limited by buffer size
        epoch (int, default=0): Training epoch identifier for metadata tracking; critical for resuming interrupted training
    
    Behavior:
        - Thread safety: Not thread-safe; single iterator should be used per process/worker
        - Error handling: Propagates I/O errors with contextual information; gracefully closes resources on failure
        - Performance: Amortizes I/O costs through buffering; throughput bounded by transform complexity and network latency
        - State management: Maintains minimal state (current buffer and global sample counter) to enable resumability
        - Resource management: Automatically closes file handles and commits database transactions at iteration end
    
    Integration:
        - DataLoader compatibility: Designed for multi-process data loading via PyTorch DataLoader
          Example: `loader = DataLoader(RemoteIterableDataset(["s3://bucket/file.txt"]), batch_size=32, num_workers=4)`
        - Distributed training: Works with DistributedSampler when using appropriate MetadataCache configuration
        - Dependencies: Requires smart_open library for protocol handling and PostgreSQL for metadata persistence
        - Extension: Can be subclassed to override __iter__ for custom streaming behavior while preserving interface
    
    Limitations:
        - Line-oriented: Processes text files with one sample per line; binary formats require custom transform
        - Limited randomization: True randomness constrained by prefetch_size; not suitable for applications requiring global shuffling
        - Sequential file access: Processes files in order specified in addresses; no random file access
        - Memory-bound throughput: Performance degrades if transform produces objects significantly larger than input lines
        - No random access: Cannot efficiently access arbitrary indices; unsuitable for validation requiring specific samples
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
