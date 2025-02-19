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
    Pre-process raw data files from input_dir and package them into a tar archive.

    The output tar archive will include all files found within the specified
    input directory, preserving the relative directory structure.
    Although this function currently does not apply any transformation on the files,
    it is structured to allow future pre-processing steps.

    Args:
        input_dir (str): Path to the directory containing raw data files.
        output_tar_path (str): Path where the output tar archive will be created.
    
    Raises:
        ValueError: If the input directory does not exist or is not a directory.
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
    Upload the specified file to an AWS S3 bucket.

    Args:
        file_path (str): Path to the file to upload.
        bucket (str): Name of the S3 bucket.
        object_key (str): Key (path in bucket) to place the uploaded file.
    
    Raises:
        Exception: If the upload fails.
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