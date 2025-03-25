#!/usr/bin/env python
"""
Example script to demonstrate training a Vorox model using a configuration file.
"""

import argparse
import logging
import yaml
import os
from pathlib import Path

from vorox.configs import RunConfig
from vorox.train import train

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """
    Main function to parse arguments and start training.
    """
    parser = argparse.ArgumentParser(description="Train a Vorox model using a configuration file")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/20M_test_model.yml",
        help="Path to the configuration YAML file"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        choices=["cpu", "cuda", "mps"], 
        help="Override device specified in config"
    )
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        help="Path to checkpoint to resume training from"
    )
    parser.add_argument(
        "--wandb_project", 
        type=str, 
        help="Override W&B project name"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        help="Override checkpoint directory"
    )
    
    args = parser.parse_args()
    
    # Check if config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Load configuration
    logger.info(f"Loading configuration from {config_path}")
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    
    # Override config with command line arguments if provided
    if args.device:
        config_dict["hardware"]["device"] = args.device
        logger.info(f"Overriding device with: {args.device}")
    
    if args.checkpoint:
        config_dict["checkpoint"]["load_from_checkpoint"] = args.checkpoint
        logger.info(f"Setting checkpoint to load from: {args.checkpoint}")
    
    if args.wandb_project:
        config_dict["logging"]["wandb_project"] = args.wandb_project
        logger.info(f"Overriding W&B project with: {args.wandb_project}")
    
    if args.output_dir:
        config_dict["checkpoint"]["checkpoint_dir"] = args.output_dir
        logger.info(f"Overriding checkpoint directory with: {args.output_dir}")
        # Ensure the output directory exists
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Create RunConfig from dictionary
    cfg = RunConfig.model_validate(config_dict)
    
    # Start training
    logger.info(f"Starting training with experiment name: {cfg.experiment_name}")
    results = train(cfg)
    
    # Log results
    logger.info("Training completed!")
    if results["best_checkpoint_path"]:
        logger.info(f"Best checkpoint saved at: {results['best_checkpoint_path']}")
        logger.info(f"Best metric value: {results['best_metric_value']}")
    
    logger.info(f"Final loss: {results['final_loss']}")
    
    return results

if __name__ == "__main__":
    main()
