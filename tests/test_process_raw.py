import os
import tarfile
import tempfile
import boto3
import pytest
from moto import mock_s3
from data.process_raw import preprocess_files, upload_to_s3

def test_preprocess_files_creates_tar_archive():
    """
    Test that verifies the preprocess_files function creates a tar archive containing
    all files from the input directory.
    """
    with tempfile.TemporaryDirectory() as temp_input_dir:
        # Create dummy files in the temporary directory.
        file_names = []
        for i in range(3):
            filename = f"file_{i}.txt"
            file_path = os.path.join(temp_input_dir, filename)
            with open(file_path, "w") as f:
                f.write(f"Dummy content for file {i}")
            file_names.append(filename)

        # Path for the temporary tar output (inside the same temporary directory)
        output_tar = os.path.join(temp_input_dir, "output.tar")
        preprocess_files(temp_input_dir, output_tar)

        # Verify the tar archive was created and includes the dummy files.
        with tarfile.open(output_tar, "r") as tar:
            members = tar.getnames()
            for name in file_names:
                assert name in members, f"{name} not found in tar archive"

@mock_s3
def test_upload_to_s3():
    """
    Test that verifies the upload_to_s3 function correctly uploads a file to a simulated S3 bucket.
    The test uses the moto library to simulate S3.
    """
    region = "us-east-1"
    bucket_name = "test-bucket"
    object_key = "uploaded/output.tar"

    # Set up a simulated S3 bucket.
    s3_client = boto3.client("s3", region_name=region)
    s3_client.create_bucket(Bucket=bucket_name)

    # Create a temporary file to act as the tar file.
    with tempfile.NamedTemporaryFile("w", delete=False) as temp_file:
        temp_file.write("Dummy content for S3 upload test")
        temp_file_path = temp_file.name

    try:
        upload_to_s3(temp_file_path, bucket_name, object_key)
        # Verify that the file now exists in the simulated S3 bucket.
        response = s3_client.list_objects_v2(Bucket=bucket_name)
        keys = [obj["Key"] for obj in response.get("Contents", [])]
        assert object_key in keys, f"{object_key} not found in bucket {bucket_name}"
    finally:
        os.remove(temp_file_path) 