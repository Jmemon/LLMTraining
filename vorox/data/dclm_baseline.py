from huggingface_hub import HfApi, hf_hub_download
import boto3
import os
from dotenv import load_dotenv
from pathlib import Path
import tempfile
from tqdm import tqdm
import zstandard as zstd
import json
import tarfile
import io

DCLM_PREFIX = "dclm-baseline"
DCLM_BASELINE_REPO_ID = "mlfoundations/dclm-baseline-1.0"


def get_dclm_baseline_urls(bucket: str, prefix: str = DCLM_PREFIX) -> list[str]:
    s3_client = boto3.client('s3')
    response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
    return [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].startswith(prefix)]


def convert_jsonl_zst_to_webdataset(input_path: str) -> io.BytesIO:
    """
    Converts a .jsonl.zst file to a WebDataset-format tar file.
    Each JSON object is split into:
    - {index}.txt: Contains the main text content
    - {index}.json: Contains the metadata
    
    Args:
        input_path (str): Path to the input .jsonl.zst file
    
    Returns:
        io.BytesIO: A buffer containing the WebDataset-format tar file
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
    Downloads .jsonl.zst files from a Hugging Face repository, converts them to WebDataset-format tar files,
    and uploads them to S3.
    
    Args:
        repo_id (str): The Hugging Face repository ID
        bucket (str): The S3 bucket name to upload files to
        num_files (int): Maximum number of files to process
        prefix (str): Optional prefix for S3 keys
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


if __name__ == "__main__":
    download_convert_and_upload_hf_to_s3("mlfoundations/dclm-baseline-1.0", "vorox-processed-train-data", 10, "dclm-baseline")
    print(get_dclm_baseline_urls("vorox-processed-train-data", "dclm-baseline"))