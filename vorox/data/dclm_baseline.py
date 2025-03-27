from datasets import load_dataset
import torch
from torch.utils.data import IterableDataset

DCLM_PREFIX = "dclm-baseline"
DCLM_BASELINE_REPO_ID = "mlfoundations/dclm-baseline-1.0"

class DCLMBaselineDataset(IterableDataset):
    """
    PyTorch IterableDataset wrapper for DCLM Baseline dataset.
    
    Ensures compatibility with PyTorch DataLoader's distributed training features
    by properly handling worker initialization and data sharding.
    """
    def __init__(self):
        self.dataset = load_dataset(DCLM_BASELINE_REPO_ID, split='train', streaming=True)
    
    def __len__(self):
        """
        Returns the number of items in the dataset.
        """
        return len(self.dataset)
        
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        
        # If we have multiple workers, shard the dataset
        if worker_info is not None:
            # Split dataset based on worker id and total workers
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            
            # Skip samples that aren't for this worker
            dataset_iter = iter(self.dataset)
            for i, sample in enumerate(dataset_iter):
                try:
                    if i % num_workers == worker_id:
                        yield {"text": sample["text"]}
                except Exception as e:
                    print(f"Error processing sample: {sample}")
                    print(f"Exception: {e}")
                    raise
        else:
            # Single worker case - process all samples
            for sample in self.dataset:
                try:
                    yield {"text": sample["text"]}
                except Exception as e:
                    print(f"Error processing sample: {sample}")
                    print(f"Exception: {e}")
                    raise

def dclm_baseline():
    """
    Returns a PyTorch-compatible dataset for the DCLM baseline.
    
    The returned dataset is compatible with PyTorch DataLoader's distributed
    training capabilities (num_workers > 1) and properly handles data sharding.
    
    Returns:
        DCLMBaselineDataset: A PyTorch IterableDataset implementation that:
        - Converts data to PyTorch tensors
        - Supports multiple workers in DataLoader
        - Properly shards data across workers
    """
    return DCLMBaselineDataset()

