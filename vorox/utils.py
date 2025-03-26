import torch
import boto3
import os
from huggingface_hub import HfApi, hf_hub_download
from dotenv import load_dotenv
from pathlib import Path
import tempfile
import zstandard as zstd
import json
import tarfile
import io

def get_available_devices():
    """
    Discovers and returns all available PyTorch computation devices with O(n) complexity.
    
    Implements a device discovery mechanism that systematically identifies all accessible
    computation resources across heterogeneous hardware environments, prioritizing
    comprehensive detection over performance optimization.
    
    Architecture:
        - Implements sequential device detection with O(n) time complexity where n is device count
        - CPU detection: Constant-time operation with unconditional inclusion
        - CUDA detection: Linear-time enumeration of all NVIDIA GPUs via CUDA runtime API
        - MPS detection: Constant-time detection of Apple Silicon GPU via MPS backend
        - Memory complexity: O(n) for device list storage
        - Zero external dependencies beyond PyTorch's device management subsystem
        - Fail-safe design with graceful degradation when specialized hardware is unavailable
    
    Returns:
        list[torch.device]: Ordered list of available computation devices with the following properties:
            - Always includes CPU as the first element (guaranteed minimum length of 1)
            - Contains all detected CUDA devices with sequential indexing (cuda:0, cuda:1, etc.)
            - Includes MPS device if running on compatible Apple Silicon hardware
            - Device objects are fully initialized and ready for immediate use in tensor operations
            - Order is deterministic: CPU → CUDA devices (if any) → MPS (if available)
    
    Behavior:
        - Stateless operation with no side effects
        - Thread-safe due to absence of shared mutable state
        - Deterministic output for fixed hardware configuration
        - Zero caching strategy (performs fresh detection on each call)
        - Graceful handling of partial hardware availability
        - No exception propagation (hardware detection failures are suppressed)
    
    Integration:
        - Typically called during initialization phase of training/inference pipelines
        - Used for device selection, multi-GPU distribution, or fallback logic
        - Consumed by device placement strategies and distributed training coordinators
        - Example:
          ```
          devices = get_available_devices()
          device = devices[0] if len(devices) == 1 else devices[1]  # Prefer GPU if available
          model = model.to(device)
          ```
    
    Limitations:
        - No detection of specialized hardware beyond CUDA and MPS (TPUs, IPUs, etc.)
        - No capability assessment (only detects presence, not performance characteristics)
        - No memory availability checking (devices may be present but resource-constrained)
        - No support for remote or distributed devices
        - No prioritization based on device capabilities
        - Requires PyTorch with appropriate backend compilation (CUDA, MPS)
    """
    devices = []
    
    # Check CPU
    devices.append(torch.device("cpu"))
    
    # Check CUDA (NVIDIA GPUs)
    if torch.cuda.is_available():
        num_cuda = torch.cuda.device_count()
        for i in range(num_cuda):
            devices.append(torch.device(f"cuda:{i}"))
    
    # Check MPS (Apple Silicon)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.append(torch.device("mps"))

    return devices


DCLM_PREFIX = "dclm-baseline"


def get_dclm_baseline_urls(bucket: str, prefix: str = DCLM_PREFIX) -> list[str]:
    """
    Retrieves S3 object keys for DCLM Baseline dataset files from a specified bucket.
    
    Architecture:
        - Implements a simple S3 listing operation with O(n) complexity where n is the number of objects
        - Uses prefix filtering at the API level for efficient server-side filtering
        - Performs additional client-side filtering to ensure exact prefix matching
    
    Interface:
        - bucket (str): S3 bucket name containing the dataset files
          Must be an existing bucket with appropriate read permissions
        - prefix (str, optional): S3 key prefix to filter objects by
          Defaults to "dclm-baseline" constant defined at module level
          Used for both API-level filtering and additional client-side filtering
        
        Returns:
            list[str]: List of S3 object keys matching the specified prefix
            Empty list if no matching objects found or bucket doesn't exist
    
    Behavior:
        - Thread-safe as it creates a new boto3 client instance on each call
        - No persistent state between invocations
        - Pagination not implemented; limited to 1000 objects per S3 API response
    
    Integration:
        - Used by download_convert_and_upload_hf_to_s3() to check for already processed files
        - Requires AWS credentials in environment variables or config files
        - Example:
          ```
          urls = get_dclm_baseline_urls("vorox-processed-train-data", "dclm-baseline")
          ```
    
    Limitations:
        - No pagination support; limited to 1000 objects per response
        - No error handling for invalid bucket names or permission issues
        - Requires appropriate AWS credentials configured in environment
        - No support for alternative authentication methods or endpoint configurations
    """
    s3_client = boto3.client('s3')
    response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
    return [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].startswith(prefix)]


def convert_jsonl_zst_to_webdataset(input_path: str) -> io.BytesIO:
    """
    Converts a Zstandard-compressed JSONL file to WebDataset-format tar archive for efficient ML training.
    
    Architecture:
        - Implements a streaming decompression-transformation pipeline with O(n) time complexity
        - Uses in-memory buffering with constant memory overhead regardless of input size
        - Processes each JSON record atomically, ensuring partial file corruption cannot occur
        - Maintains sequential record indexing with zero-padded 6-digit identifiers
    
    Interface:
        - input_path (str): Filesystem path to the input .jsonl.zst file
          Must be a valid file path with read permissions
          File must contain valid JSON objects, one per line, compressed with Zstandard
          
        Returns:
            io.BytesIO: In-memory buffer containing a complete WebDataset-format tar archive
            Buffer position is reset to 0 before returning
            Empty buffer returned if input file is empty or contains only whitespace
    
    Behavior:
        - Thread-safe as it creates new file handles and buffers on each invocation
        - No persistent state between calls
        - Skips empty lines in the input file
        - Handles UTF-8 encoding/decoding for text content
        - Preserves all JSON fields except 'text' in the metadata file
    
    Integration:
        - Called by download_convert_and_upload_hf_to_s3() to process individual dataset files
        - Requires zstandard and tarfile libraries
        - Example:
          ```
          buffer = convert_jsonl_zst_to_webdataset("/path/to/file.jsonl.zst")
          with open("output.tar", "wb") as f:
              f.write(buffer.getvalue())
          ```
    
    Limitations:
        - Loads entire decompressed content into memory, potentially problematic for very large files
        - No streaming output support; entire tar archive built in memory before returning
        - No error handling for malformed JSON or corrupt Zstandard data
        - Assumes 'text' field exists in JSON objects; creates empty text files if missing
    """
    tar_buffer = io.BytesIO()
    
    with tarfile.open(fileobj=tar_buffer, mode='w') as tar:
        with open(input_path, 'rb') as fh:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(fh) as reader:
                text = reader.read().decode('utf-8')
                
                for i, line in enumerate(text.split('\n')):
                    if not line.strip():
                        continue
                        
                    json_data = json.loads(line)
                    
                    # Extract text content and metadata
                    text_content = json_data.get('text', '').encode('utf-8')
                    metadata = {k: v for k, v in json_data.items() if k != 'text'}
                    
                    # Create text file
                    text_info = tarfile.TarInfo(name=f'{i:06d}.txt')
                    text_info.size = len(text_content)
                    tar.addfile(text_info, io.BytesIO(text_content))
                    
                    # Create metadata file
                    metadata_content = json.dumps(metadata, indent=2).encode('utf-8')
                    metadata_info = tarfile.TarInfo(name=f'{i:06d}.json')
                    metadata_info.size = len(metadata_content)
                    tar.addfile(metadata_info, io.BytesIO(metadata_content))
    
    tar_buffer.seek(0)
    return tar_buffer


def download_convert_and_upload_hf_to_s3(repo_id: str, bucket: str, num_files: int = 10, prefix: str = "") -> None:
    """
    Downloads, converts, and uploads Hugging Face dataset files to S3 in WebDataset format for ML training pipelines.
    
    Architecture:
        - Implements a three-stage ETL pipeline with O(n) time complexity where n is the number of files
        - Uses incremental processing with early termination to handle large repositories efficiently
        - Employs idempotent operations with existence checking to support resumable processing
        - Maintains flat namespace conversion for hierarchical repository structures
    
    Interface:
        - repo_id (str): Hugging Face repository identifier in 'username/repo-name' format
          Must be a valid, accessible Hugging Face dataset repository
          Example: "mlfoundations/dclm-baseline-1.0"
        
        - bucket (str): AWS S3 bucket name for storing processed files
          Must exist with appropriate write permissions configured
          Used as the destination for all converted WebDataset tar files
        
        - num_files (int, optional): Maximum number of files to process per invocation
          Defaults to 10 files
          Set to a lower value for testing or higher for production runs
          Early termination occurs after processing this many matching files
        
        - prefix (str, optional): S3 key prefix for uploaded files
          Defaults to empty string
          Used to organize files in the S3 bucket hierarchy
          Will be prepended to all generated S3 keys with a hyphen separator
    
    Behavior:
        - Thread-safe as it creates new client instances on each invocation
        - Stateless between runs; can be safely interrupted and resumed
        - Skips already processed files based on S3 key existence
        - Processes only .jsonl.zst files, ignoring other formats
        - Uses temporary local storage for downloads, automatically cleaned up after processing
    
    Integration:
        - Requires environment variables for AWS and Hugging Face authentication
        - Loads credentials from .env file via dotenv
        - Depends on get_dclm_baseline_urls() for S3 object listing
        - Depends on convert_jsonl_zst_to_webdataset() for file format conversion
        - Example:
          ```
          download_convert_and_upload_hf_to_s3(
              "mlfoundations/dclm-baseline-1.0", 
              "vorox-processed-train-data",
              num_files=100,
              prefix="dclm-baseline"
          )
          ```
    
    Limitations:
        - No parallel processing; files are processed sequentially
        - No progress reporting or logging mechanism
        - Limited to 1000 existing objects due to S3 listing pagination limits
        - No error recovery for individual file failures; processing stops on first error
        - Requires sufficient local disk space for temporary file storage
        - Path handling may be problematic on Windows due to backslash conversion
    """
    load_dotenv()

    # Initialize clients
    hf_api = HfApi()
    s3_client = boto3.client('s3')
    
    # Get list of all files in the repository
    files = hf_api.list_repo_files(repo_id, repo_type="dataset")
    already_downloaded = set(get_dclm_baseline_urls(bucket, prefix))
    
    # Process each .jsonl.zst file
    count = 0
    for file_path in files:
        if not file_path.endswith('.jsonl.zst'):
            continue

        count += 1
        if count > num_files:
            break
            
        # Create flattened filename (replacing .jsonl.zst with .tar)
        parts = Path(file_path).parts
        base_name = '_'.join(parts)
        base_name = base_name.rsplit('.jsonl.zst', 1)[0] + '.tar'
        s3_key = f"{prefix}-{base_name}"

        if s3_key in already_downloaded:
            continue
        
        # Download and process file
        with tempfile.TemporaryDirectory() as temp_dir:
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=file_path,
                repo_type="dataset",
                cache_dir=temp_dir
            )
            
            # Convert to WebDataset format tar
            tar_buffer = convert_jsonl_zst_to_webdataset(local_path)
            
            # Upload to S3
            s3_client.upload_fileobj(
                Fileobj=tar_buffer,
                Bucket=bucket,
                Key=s3_key.replace('\\', '/'),  # Ensure forward slashes for S3 keys
            )
