"""
This script loads the configuration, initializes the tokenizer, model, optimizer, loss function,
and constructs a training dataset using WebDataset. The dataset streams data from remote tar files 
(according to the URLs in the config's data.source.urls field). WebDataset automatically handles 
prefetching, shuffling, and decoding, and it can be combined with multi-worker DataLoader for 
parallel data loading.

Prerequisites:
- The configuration file must supply the "data" section with fields such as:
    data:
      settings:
        prefetch_size: <int>      # used here for shuffling buffer size
      source:
        urls:
          - "s3://bucket-name/path/to/data1.tar"
          - "s3://bucket-name/path/to/data2.tar"
- Each tar file is expected to contain samples with a key (here "txt") holding the text data.
- You can further customize the .decode, .to_tuple, etc., as needed.
"""

from dotenv import load_dotenv
import yaml
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import logging
import webdataset as wds
import boto3
import requests
from urllib.parse import urlparse

from vorox.configs import Config
from vorox.loss import LossBase
from vorox.optimizer import OptimizerBase
from vorox.train import fit
from vorox.vorox import Vorox

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

def verify_urls(urls):
    """
    Verify that each URL is accessible.
    
    Args:
        urls (List[str]): List of S3 URLs to verify
        
    Raises:
        Exception: If any URL is inaccessible
    """
    for url in urls:
        try:
            # Parse the S3 URL
            parsed = urlparse(url)
            
            # Extract bucket name - handle both virtual-hosted and path-style URLs
            if '.s3.' in parsed.netloc:
                # Virtual-hosted style URL (e.g., bucket-name.s3.region.amazonaws.com)
                bucket = parsed.netloc.split('.s3.')[0]
            else:
                # Path-style URL (e.g., s3://bucket-name/...)
                bucket = parsed.netloc
            
            # Remove leading slash and decode URL-encoded characters
            key = parsed.path.lstrip('/')
            
            logging.info(f"Attempting to access - Bucket: {bucket}, Key: {key}")
            
            # Try to get the head of the object
            s3 = boto3.client('s3')
            response = s3.head_object(Bucket=bucket, Key=key)
            
            logging.info(f"Successfully verified access to s3://{bucket}/{key}")
            logging.info(f"Object size: {response['ContentLength']} bytes")
            
        except Exception as e:
            logging.error(f"Failed to access {url}")
            logging.error(f"Parsed bucket: {bucket}, key: {key}")
            logging.error(f"Error details: {str(e)}")
            raise

def get_s3_url_with_credentials(url):
    """
    Convert s3:// URL to a signed HTTPS URL with temporary credentials.
    
    Args:
        url (str): S3 URL in format s3://bucket-name/key
        
    Returns:
        str: Signed HTTPS URL with temporary credentials
    """
    parsed = urlparse(url)
    
    # Extract bucket name - handle both virtual-hosted and path-style URLs
    if '.s3.' in parsed.netloc:
        # Virtual-hosted style URL (e.g., bucket-name.s3.region.amazonaws.com)
        bucket = parsed.netloc.split('.s3.')[0]
    else:
        # Path-style URL (e.g., s3://bucket-name/...)
        bucket = parsed.netloc
        
    key = parsed.path.lstrip('/')
    
    s3_client = boto3.client('s3')
    
    # Generate presigned URL without needing bucket location
    signed_url = s3_client.generate_presigned_url(
        'get_object',
        Params={
            'Bucket': bucket,
            'Key': key
        },
        ExpiresIn=3600,
        HttpMethod='GET'
    )
    
    logging.info(f"Generated signed URL for bucket: {bucket}, key: {key}")
    return signed_url

def main():
    # Load configuration from YAML using pydantic.
    with open("configs/20M_test_model.yml", "r") as f:
        cfg = Config.model_validate(yaml.safe_load(f))
    
    from vorox.evaluators.builder import EvaluatorBuilder
    
    load_dotenv()
    
    # Add this test
    try:
        s3 = boto3.client('s3')
        # Extract bucket and key from first URL
        url = cfg.data.urls[0]
        bucket = url.split('/')[2].split('.')[0]
        key = '/'.join(url.split('/')[3:])
        s3.head_object(Bucket=bucket, Key=key)
        logging.info("Successfully verified S3 access")
    except Exception as e:
        logging.error(f"Failed to access S3: {str(e)}")
        raise
    
    # Initialize the tokenizer from a pretrained model as specified in the config.
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer.name)
    
    # Build the model, optimizer, and loss function.
    model = Vorox(cfg, tokenizer.vocab_size).to(cfg.device)
    optimizer = OptimizerBase.build(model, cfg)
    loss_fn = LossBase.build(cfg)
    
    # Add this call before setting up the dataset
    verify_urls(cfg.data.urls)
    
    # Convert S3 URLs to signed HTTPS URLs
    signed_urls = []
    for url in cfg.data.urls:
        try:
            signed_url = get_s3_url_with_credentials(url)
            signed_urls.append(signed_url)
            logging.info(f"Generated signed URL for {url}")
        except Exception as e:
            logging.error(f"Failed to generate signed URL for {url}: {str(e)}")
            raise

    # Set up the dataset using WebDataset with signed URLs
    dataset = (
        wds.WebDataset(
            signed_urls,
            handler=wds.handlers.warn_and_continue,
            # Add explicit empty_check=False to handle potential empty shards
            empty_check=False,
            # Add verbose error reporting
            verbose=True,
        )
        .shuffle(cfg.data.settings.prefetch_size)
        .to_tuple("txt", "json")
        .map(lambda x: tokenizer(
            x[0].decode("utf-8"), padding='max_length', truncation=True, max_length=cfg.train.max_seq_len, return_tensors="pt").input_ids[0])
    )

    # Add more detailed error checking
    try:
        # Try to get first item to verify dataset is not empty
        dataset_iterator = iter(dataset)
        try:
            first_item = next(dataset_iterator)
            logging.info("Successfully loaded first item from dataset")
            logging.info(f"First item shape/type: {first_item.shape}/{type(first_item)}")
        except StopIteration:
            # Add detailed diagnostics
            logging.error("Dataset is empty - attempting diagnosis:")
            for url in signed_urls:
                try:
                    response = requests.head(url, timeout=10)
                    logging.info(f"URL check: {url.split('?')[0]} - Status: {response.status_code}")
                except Exception as e:
                    logging.error(f"Failed to check URL {url.split('?')[0]}: {str(e)}")
            raise ValueError("Dataset is empty - please verify data files contain valid samples")
    except Exception as e:
        logging.error(f"Error checking dataset: {str(e)}")
        raise

    # The DataLoader wraps the WebDataset.
    # Explicitly set num_workers to 1 for debugging
    train_loader = DataLoader(
        dataset,
        #num_workers=cfg.data.settings.num_workers if hasattr(cfg.data.settings, "num_workers") else 4,
        num_workers=1,
        batch_size=cfg.train.macro_batch_size,
    )

    # Validation loader can also be set up similarly.
    val_loader = None  # Replace with a similar pipeline if validation data exists.
    
    # Build evaluators
    evaluators = EvaluatorBuilder.build(cfg.eval, cfg.eval.evaluators)
    
    # Start training.
    fit(cfg, model, train_loader, val_loader, optimizer, loss_fn)

if __name__ == "__main__":
    main()
