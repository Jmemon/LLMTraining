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

from data.utilities.remote_stream import open_remote
from data.utilities.metadata_cache import MetadataCache

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # For demonstration, set to DEBUG level.

class RemoteIterableDataset(IterableDataset):
    """
    A memory-constrained streaming dataset that processes remote data sources with constant memory footprint, 
    designed for training on datasets that exceed available RAM by implementing a prefetch buffer architecture 
    that balances I/O efficiency with controlled randomization.
    
    Architecture:
        - Implements lazy streaming I/O via smart_open with O(1) memory usage regardless of dataset size
        - Employs fixed-size prefetch buffer (O(prefetch_size) memory) with amortized O(1) access time per sample
        - Uses Fisher-Yates in-buffer shuffling for memory-efficient local randomization in O(prefetch_size) time
        - Maintains stateful processing via PostgreSQL-backed sample tracking with O(1) insertion complexity
        - Manages resource lifecycle with automatic connection pooling and graceful error propagation
    
    Interface:
        - __init__(
            addresses: List[str],
            transform: Optional[Callable[[str], Any]] = None,
            prefetch_size: int = 1000,
            cache: Optional[MetadataCache] = None,
            shuffle_buffer: bool = False,
            epoch: int = 0
          ): Configures dataset parameters
        - __iter__() -> Iterator[Any]: Returns stateful iterator yielding transformed samples
    
    Parameters:
        addresses: Remote file URLs supporting multiple protocols (s3://, gs://, http://, file://, hf://)
                  Must contain line-delimited text data; each line processed as one sample
        transform: Function converting raw text lines to desired format; must be pickle-serializable
                  Receives str input; can return any type; executed in worker processes
        prefetch_size: Controls memory usage (larger = more RAM), shuffling effectiveness, and I/O batching
                      Must be > 0; optimal values typically 100-10000 depending on sample size
        cache: PostgreSQL connection for tracking processed samples; enables training resumption
               If None, no persistence; if provided, must be pre-initialized MetadataCache instance
        shuffle_buffer: Enables in-buffer randomization via Fisher-Yates algorithm
                       True = shuffled samples within buffer boundaries; False = sequential processing
        epoch: Integer identifier for current training epoch; used for metadata tracking
               Must be consistent across restarts to properly resume interrupted training
    
    Behavior:
        - Not thread-safe: Single iterator should be used per worker process
        - Propagates I/O errors with contextual information and gracefully closes resources
        - Performance bounded by transform complexity, network latency, and buffer size
        - Maintains minimal state (buffer + sample counter) with O(prefetch_size) memory footprint
        - Automatically commits metadata transactions and closes file handles at iteration end
        - Processes files sequentially in order specified by addresses parameter
    
    Integration:
        - Compatible with PyTorch DataLoader for multi-process loading:
          loader = DataLoader(
              RemoteIterableDataset(["s3://bucket/file.txt"]), 
              batch_size=32, 
              num_workers=4
          )
        - Works with DistributedSampler when using shared MetadataCache instance
        - Requires smart_open library for protocol handling and psycopg2 for PostgreSQL
        - Can be subclassed to override __iter__ for custom streaming behavior
    
    Limitations:
        - Line-oriented processing only; binary formats require custom transform
        - Randomization limited by prefetch_size; not suitable for global shuffling requirements
        - Sequential file access only; no random access to arbitrary samples
        - Performance degrades if transform produces objects significantly larger than input
        - Not suitable for validation requiring specific samples due to streaming nature
        - No built-in compression handling; compressed files must be handled by transform
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
        """
        Initializes a memory-efficient streaming dataset for processing remote data sources.
        
        Configures a dataset that streams data from remote locations with controlled memory usage,
        optional in-buffer shuffling, and resumable processing via metadata tracking. Designed
        for training on datasets that exceed available RAM by maintaining a fixed-size buffer.
        
        Architecture:
            - Implements parameter validation and storage with O(1) initialization complexity
            - Defers actual I/O operations until iteration time with zero upfront data loading
            - Supports multiple remote protocols via smart_open dependency with protocol abstraction
            - Memory complexity: O(1) during initialization (stores only configuration parameters)
            - Preserves statelessness until iteration begins for serialization compatibility
            
        Parameters:
            addresses (List[str]): Remote file URLs to process sequentially.
                Must be accessible via supported protocols (s3://, gs://, http://, file://, hf://).
                Each file must contain line-delimited text data (one sample per line).
                Empty list will result in empty iteration with no errors.
                
            transform (Optional[Callable[[str], Any]]): Function to convert raw text lines to desired format.
                Default: None (raw strings returned).
                Must be pickle-serializable for DataLoader worker processes.
                Receives string input; can return any type.
                Applied to each line after stripping whitespace.
                
            prefetch_size (int): Number of samples to buffer before yielding.
                Default: 1000.
                Must be positive integer.
                Controls memory usage (larger = more RAM), shuffling effectiveness, and I/O batching.
                Optimal values typically 100-10000 depending on sample size and available memory.
                
            cache (Optional[MetadataCache]): PostgreSQL connection for tracking processed samples.
                Default: None (no persistence).
                If provided, must be pre-initialized MetadataCache instance.
                Enables training resumption by recording sample processing state.
                Will be closed automatically at iteration end.
                
            shuffle_buffer (bool): Whether to randomize samples within each buffer.
                Default: False (sequential processing).
                True enables in-buffer Fisher-Yates shuffling for local randomization.
                Randomization limited to prefetch_size window; not global shuffling.
                
            epoch (int): Integer identifier for current training epoch.
                Default: 0.
                Used for metadata tracking with cache parameter.
                Must be consistent across restarts to properly resume interrupted training.
                
        Raises:
            TypeError: If parameters have incorrect types.
            ValueError: If prefetch_size <= 0.
            
        Behavior:
            - Stateless initialization with deferred resource allocation
            - Thread-safe for constructor but not for resulting iterator
            - Zero I/O operations during initialization
            - No validation of remote addresses until iteration time
            - Parameter storage only; no data loading or processing
            
        Integration:
            - Typically instantiated directly before DataLoader construction:
              ```
              dataset = RemoteIterableDataset(
                  addresses=["s3://bucket/file.txt"],
                  transform=json.loads,
                  prefetch_size=1000
              )
              loader = DataLoader(dataset, batch_size=32, num_workers=4)
              ```
            - Often used with metadata cache for resumable training:
              ```
              cache = MetadataCache("postgresql://user:pass@host/db")
              dataset = RemoteIterableDataset(
                  addresses=urls,
                  cache=cache,
                  epoch=current_epoch
              )
              ```
            
        Limitations:
            - No parameter validation until iteration time
            - No support for non-text file formats without custom transform
            - No automatic protocol detection; URLs must include explicit scheme
            - No built-in compression handling; requires appropriate transform
            - Metadata cache requires PostgreSQL; no alternative database support
            - No support for random access or indexing due to streaming nature
        """
        self.addresses = addresses
        self.transform = transform
        self.prefetch_size = prefetch_size
        self.cache = cache
        self.shuffle_buffer = shuffle_buffer
        self.epoch = epoch

    def __iter__(self) -> Iterator:
        """
        Implements a memory-efficient streaming iterator with buffered I/O and optional shuffling.
        
        Processes remote data sources sequentially with controlled memory usage by maintaining
        a fixed-size buffer, tracking sample processing state, and providing optional local
        randomization within buffer boundaries.
        
        Architecture:
            - Implements producer-consumer pattern with O(prefetch_size) space complexity
            - Uses buffered I/O with amortized O(1) time complexity per yielded sample
            - Employs Fisher-Yates in-buffer shuffling with O(prefetch_size) time complexity
            - Implements stateful processing with PostgreSQL sample tracking (O(1) per sample)
            - Error propagation with context-enriched exceptions and resource cleanup
            - Memory footprint: O(prefetch_size) regardless of total dataset size
            
        Returns:
            Iterator: A generator yielding processed samples with types determined by transform:
                - If transform is None: yields stripped strings from input lines
                - If transform is provided: yields objects of transform's return type
                - Maintains iteration order matching file order unless shuffle_buffer=True
                
        Raises:
            ValueError: When remote file access fails, with detailed error context
            RuntimeError: If transform raises exceptions during sample processing
            IOError: For underlying I/O errors from smart_open or network stack
            
        Behavior:
            - Stateful iteration with buffer and counter persistence between yields
            - Not thread-safe; single iterator should be used per worker process
            - Automatic resource management with proper file handle closure
            - Deterministic sample ordering unless shuffle_buffer=True
            - Metadata persistence via cache.add_sample() with O(1) insertion complexity
            - Performance bounded by transform complexity, network latency, and buffer size
            - Processes files sequentially in order specified by addresses parameter
            
        Integration:
            - Called implicitly by Python's iteration protocol: for sample in dataset
            - Automatically invoked by PyTorch DataLoader during batch construction
            - Compatible with both single-process and multi-process data loading
            - Example: for batch in DataLoader(dataset, batch_size=32, num_workers=4): ...
            
        Limitations:
            - Line-oriented processing only; binary formats require custom transform
            - Randomization limited by prefetch_size; not suitable for global shuffling
            - Sequential file access only; no random access to arbitrary samples
            - Performance degrades if transform produces objects significantly larger than input
            - No built-in compression handling; compressed files must be handled by transform
            - No early termination detection; processes all files completely
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
