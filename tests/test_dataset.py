"""
Tests for the RemoteIterableDataset. Uses a temporary file (via file:// URL) to simulate a remote file.
"""

import os
import tempfile
from data.dataset import RemoteIterableDataset
from data.metadata_cache import MetadataCache

def test_remote_iterable_dataset():
    # Create a temporary file with dummy data.
    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        f.write("sample1\nsample2\nsample3\nsample4\nsample5\n")
        temp_file = f.name

    url = f"file://{temp_file}"
    # Create a temporary metadata cache.
    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        cache_file = f.name

    cache = MetadataCache(cache_file)
    dataset = RemoteIterableDataset(
        addresses=[url],
        transform=lambda x: x.upper(),  # make samples uppercase
        prefetch_size=2,
        cache=cache,
        shuffle_buffer=False,
        epoch=1
    )
    
    samples = list(dataset)
    # Expect samples to be the uppercase versions of "sample1" ... "sample5"
    assert samples == ["SAMPLE1", "SAMPLE2", "SAMPLE3", "SAMPLE4", "SAMPLE5"]
    
    # Check that metadata cache recorded all five samples.
    epoch_samples = cache.get_samples(1)
    assert epoch_samples == list(range(5))
    
    os.remove(temp_file)
    os.remove(cache_file)

if __name__ == "__main__":
    test_remote_iterable_dataset()
    print("test_remote_iterable_dataset passed!") 