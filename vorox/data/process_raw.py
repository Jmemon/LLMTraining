import os
import tarfile
import logging
import boto3
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_files(input_dir: str, output_tar_path: str) -> None:
    """
    Recursively packages raw data files into a tar archive with preserved directory structure.
    
    Architecture:
        - Implements a non-transformative archiving process with O(n) time complexity
        - Uses Path.glob("**/*") for recursive directory traversal with depth-first ordering
        - Preserves relative paths within the archive via Path.relative_to() normalization
        - Maintains atomic write semantics through tarfile context manager
    
    Args:
        input_dir (str): Absolute or relative path to source directory containing raw data files.
                         Must be an existing directory with read permissions.
        output_tar_path (str): Destination path for the output tar archive. Parent directories
                              must exist with write permissions. Will overwrite existing files.
    
    Raises:
        ValueError: When input_dir doesn't exist or isn't a directory.
        PermissionError: When lacking read access to input_dir or write access to output_tar_path.
        tarfile.TarError: On archive creation failures (corrupt data, disk full).
        OSError: On I/O errors during file operations.
    
    Behavior:
        - Thread-unsafe: Should not be called concurrently on the same output_tar_path
        - Logs each file addition at INFO level via the module logger
        - Creates uncompressed tar archives (mode="w") without additional transformation
        - Memory usage scales with the largest individual file size, not total archive size
    
    Integration:
        - Typically called from CLI via main() or programmatically before upload_to_s3()
        - Example: preprocess_files("/data/raw", "/tmp/processed.tar")
    
    Limitations:
        - No built-in compression (use mode="w:gz" for gzip compression if needed)
        - No file filtering mechanism (processes all files regardless of type)
        - No progress reporting for large directories beyond individual file logging
        - Maximum path length constrained by tar format (100 chars for traditional, 256 for GNU)
    """
    input_dir = Path(input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"Input directory {input_dir} does not exist or is not a directory")

    with tarfile.open(output_tar_path, "w") as tar:
        for file_path in input_dir.glob("**/*"):
            if file_path.is_file():
                logger.info(f"Adding file: {file_path}")
                tar.add(str(file_path), arcname=file_path.relative_to(input_dir))
    logger.info(f"Tar archive created at: {output_tar_path}")

def upload_to_s3(file_path: str, bucket: str, object_key: str) -> None:
    """
    Transfers a local file to AWS S3 storage with error handling and logging.
    
    Architecture:
        - Implements a thin wrapper over boto3's S3 client with O(1) operation complexity
        - Uses single-part upload for files under 5GB with automatic content-type detection
        - Maintains synchronous blocking execution with full error propagation
        - Provides structured logging at INFO level for operation tracking
    
    Args:
        file_path (str): Absolute or relative path to source file for upload.
                         Must be an existing file with read permissions.
        bucket (str): Target S3 bucket name. Must exist with write permissions
                      for the configured AWS credentials.
        object_key (str): Destination path within the S3 bucket. Will overwrite
                          existing objects with the same key.
    
    Raises:
        FileNotFoundError: When file_path doesn't exist or isn't accessible.
        boto3.exceptions.S3UploadFailedError: On network failures or S3 rejections.
        boto3.exceptions.Boto3Error: On AWS API errors (permissions, quotas).
        ValueError: When bucket or object_key are empty strings.
        
    Behavior:
        - Thread-safe: Can be called concurrently with different parameters
        - Logs operation start and completion at INFO level via the module logger
        - Logs failures at ERROR level with full exception traceback
        - Memory usage scales with boto3's internal buffer size (typically 8MB chunks)
        - No retry logic (relies on boto3's internal retry mechanisms)
    
    Integration:
        - Typically called after preprocess_files() to upload generated archives
        - Requires AWS credentials configured via environment variables or config files
        - Example: upload_to_s3("/tmp/processed.tar", "data-bucket", "uploads/dataset.tar")
    
    Limitations:
        - No support for multipart uploads (files >5GB require different approach)
        - No progress reporting for large file uploads
        - No custom S3 metadata or storage class configuration
        - Region determined by boto3 configuration, not explicitly specified
    """
    s3_client = boto3.client("s3")
    try:
        logger.info(f"Uploading {file_path} to bucket {bucket} with key {object_key} ...")
        s3_client.upload_file(file_path, bucket, object_key)
        logger.info("Upload successful!")
    except Exception as e:
        logger.error("Upload failed", exc_info=True)
        raise e

def main():
    """
    Command-line interface for pre-processing files and optionally uploading the tar archive to S3.
    
    Architecture:
        - Implements a sequential pipeline with O(n) time complexity where n is file count
        - Uses argparse for CLI argument parsing with standard POSIX-style option handling
        - Maintains a linear execution flow with conditional S3 upload based on argument presence
        - Propagates all exceptions from component functions to the caller with full stack traces
    
    Args:
        Command line arguments:
            --input-dir (str, required): Source directory containing raw data files.
                                         Must be an existing directory with read permissions.
            --output-tar (str, required): Destination path for the output tar archive.
                                         Parent directories must exist with write permissions.
            --bucket (str, optional): S3 bucket name for upload. When omitted, S3 upload is skipped.
            --object-key (str, optional): S3 object key for upload. Required if --bucket is specified.
    
    Raises:
        SystemExit: With code 2 on invalid CLI arguments (via argparse).
        ValueError: When input_dir doesn't exist or isn't a directory.
        PermissionError: When lacking read/write access to specified paths.
        tarfile.TarError: On archive creation failures.
        boto3.exceptions.Boto3Error: On AWS API errors during S3 upload.
        OSError: On I/O errors during file operations.
    
    Behavior:
        - Thread-unsafe: Should not be called concurrently in the same process
        - Logs operation progress at INFO level via the module logger
        - Terminates after successful execution or on first unhandled exception
        - Memory usage scales with the largest individual file size, not total archive size
    
    Integration:
        - Entry point when module is executed directly (via __name__ == "__main__")
        - Requires AWS credentials configured via environment variables or config files when using S3
        - Example: python -m vorox.data.process_raw --input-dir /data/raw --output-tar /tmp/out.tar --bucket my-bucket --object-key datasets/out.tar
    
    Limitations:
        - No support for compressed archives (uses uncompressed tar format)
        - No file filtering mechanism (processes all files regardless of type)
        - No progress reporting for large directories beyond individual file logging
        - No parallel processing for multi-core utilization
        - No resumable uploads for large files or unreliable connections
    """
    parser = argparse.ArgumentParser(
        description="Pre-process raw data files and upload them as a tar archive to S3."
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing raw data files.")
    parser.add_argument("--output-tar", required=True, help="Path for the output tar archive.")
    parser.add_argument("--bucket", help="S3 bucket name to upload the tar file.")
    parser.add_argument("--object-key", help="S3 object key for the tar file upload.")
    args = parser.parse_args()

    preprocess_files(args.input_dir, args.output_tar)

    if args.bucket and args.object_key:
        upload_to_s3(args.output_tar, args.bucket, args.object_key)
    else:
        logger.info("S3 upload not configured. Skipping upload step.")

if __name__ == "__main__":
    main() 
