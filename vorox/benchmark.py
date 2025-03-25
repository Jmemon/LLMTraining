#!/usr/bin/env python
"""
Benchmark module for evaluating Vorox models using various evaluators.
"""

import logging
import torch
import wandb
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import time
import json

from vorox.configs import RunConfig, EvaluatorType
from vorox.evaluators.builder import EvaluatorBuilder
from vorox.vorox import Vorox
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

def benchmark(checkpoint_dir: Path, evaluators: Optional[List[EvaluatorType]] = None) -> Dict:
    """
    Benchmarks a trained Vorox model against specified evaluators.
    
    This function loads a model from a checkpoint, runs it through specified evaluators,
    and logs the results to W&B. It provides a comprehensive evaluation
    of model performance across multiple benchmarks with detailed metrics.
    
    Parameters:
        checkpoint_dir (Path): Path to the directory containing model checkpoints
        evaluators (Optional[List[EvaluatorType]]): List of evaluators to run
            If None, uses evaluators from the checkpoint's config
    
    Returns:
        Dict: Dictionary containing evaluation results:
            - 'evaluator_name': Dict of metrics for each evaluator
            - 'overall': Dict of aggregated metrics across evaluators
            - 'metadata': Dict with benchmark metadata (runtime, device, etc.)
    
    Raises:
        FileNotFoundError: If checkpoint_dir does not exist
        ValueError: If no valid checkpoints are found
    """
    # Validate checkpoint directory
    if not checkpoint_dir.exists() or not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    # Find the best checkpoint (or latest if no best exists)
    checkpoints = list(checkpoint_dir.glob("*_best.pt"))
    if not checkpoints:
        checkpoints = list(checkpoint_dir.glob("*_last.pt"))
    if not checkpoints:
        checkpoints = list(checkpoint_dir.glob("*.pt"))
    
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")
    
    # Sort by modification time (newest first)
    checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    checkpoint_path = checkpoints[0]
    
    # Start timing
    start_time = time.time()
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    except Exception as e:
        raise ValueError(f"Failed to load checkpoint: {e}")
    
    # Extract config from checkpoint
    if 'config' not in checkpoint:
        raise ValueError("Checkpoint does not contain model configuration")
    
    # Convert config dict to RunConfig
    if isinstance(checkpoint['config'], dict):
        cfg = RunConfig.model_validate(checkpoint['config'])
    else:
        # Assume it's already a RunConfig or similar
        cfg = checkpoint['config']
    
    # Determine device
    device = torch.device(cfg.hardware.device)
    logger.info(f"Using device: {device}")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.tokenizer_name)
    logger.info(f"Initialized tokenizer: {cfg.model.tokenizer_name}")
    
    # Initialize model
    model = Vorox(cfg, tokenizer.vocab_size)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    logger.info(f"Model loaded successfully with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Determine which evaluators to use
    if evaluators is None and hasattr(cfg, 'eval'):
        evaluators = cfg.eval.evaluators
    elif evaluators is None:
        # Default to all available evaluators if none specified
        evaluators = [EvaluatorType.mmlu, EvaluatorType.gsm8k, EvaluatorType.gsm_symbolic]
        logger.info(f"No evaluators specified, using defaults: {[e.value for e in evaluators]}")
    
    # Initialize W&B if enabled
    if hasattr(cfg, 'logging') and cfg.logging.wandb_project:
        project = cfg.logging.wandb_project
        entity = cfg.logging.wandb_entity if hasattr(cfg.logging, 'wandb_entity') else None
        tags = cfg.logging.wandb_tags if hasattr(cfg.logging, 'wandb_tags') else []
        
        # Add benchmark tag
        if 'benchmark' not in tags:
            tags.append('benchmark')
        
        # Extract experiment name from checkpoint
        experiment_name = cfg.experiment_name if hasattr(cfg, 'experiment_name') else "benchmark"
        run_name = f"benchmark_{experiment_name}_{checkpoint_path.stem}"
        
        wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            tags=tags,
            config={
                "checkpoint": str(checkpoint_path),
                "evaluators": [e.value for e in evaluators],
                "device": str(device),
                "model_config": cfg.model.model_dump() if hasattr(cfg, 'model') else {},
            }
        )
        logger.info(f"Initialized W&B project: {project}")
    
    # Build evaluators
    eval_config = cfg.eval if hasattr(cfg, 'eval') else None
    evaluator_instances = EvaluatorBuilder.build(eval_config, evaluators)
    logger.info(f"Built {len(evaluator_instances)} evaluators: {[e.name.value for e in evaluator_instances]}")
    
    # Run evaluations
    results = {}
    overall_metrics = {"total_correct": 0, "total_questions": 0}
    
    for evaluator in evaluator_instances:
        logger.info(f"Running evaluation with {evaluator.name}")
        try:
            eval_result = evaluator(model)
            results[evaluator.name.value] = eval_result
            
            # Log to W&B if enabled
            if wandb.run is not None:
                for metric_name, metric_value in eval_result.items():
                    wandb.log({f"eval/{evaluator.name.value}/{metric_name}": metric_value})
            
            # Update overall metrics if available
            if "num_correct" in eval_result and "num_total" in eval_result:
                overall_metrics["total_correct"] += eval_result["num_correct"]
                overall_metrics["total_questions"] += eval_result["num_total"]
            
            logger.info(f"Evaluation results for {evaluator.name}: {eval_result}")
        except Exception as e:
            logger.error(f"Error running evaluator {evaluator.name}: {e}")
            results[evaluator.name.value] = {"error": str(e)}
    
    # Calculate overall accuracy if we have totals
    if overall_metrics["total_questions"] > 0:
        overall_metrics["overall_accuracy"] = overall_metrics["total_correct"] / overall_metrics["total_questions"]
        
        # Log overall metrics to W&B
        if wandb.run is not None:
            wandb.log({"eval/overall/accuracy": overall_metrics["overall_accuracy"]})
    
    # Calculate runtime
    end_time = time.time()
    runtime = end_time - start_time
    
    # Add metadata to results
    metadata = {
        "runtime_seconds": runtime,
        "device": str(device),
        "checkpoint": str(checkpoint_path),
        "evaluators": [e.value for e in evaluators],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    # Log metadata to W&B
    if wandb.run is not None:
        wandb.log({"metadata": metadata})
        wandb.finish()
    
    # Combine all results
    final_results = {
        **results,
        "overall": overall_metrics,
        "metadata": metadata,
    }
    
    # Save results to JSON file next to checkpoint
    results_path = checkpoint_dir / f"benchmark_results.json"
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    logger.info(f"Benchmark results saved to {results_path}")
    
    return final_results

if __name__ == "__main__":
    import argparse
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Benchmark Vorox models")
    parser.add_argument(
        "--checkpoint_dir", 
        type=str, 
        required=True,
        help="Directory containing model checkpoints to benchmark"
    )
    parser.add_argument(
        "--evaluators", 
        type=str, 
        nargs="+",
        choices=[e.value for e in EvaluatorType],
        help="Evaluators to run (default: all)"
    )
    
    args = parser.parse_args()
    
    # Convert evaluator strings to EvaluatorType enums if provided
    eval_types = None
    if args.evaluators:
        eval_types = [EvaluatorType(e) for e in args.evaluators]
    
    # Run benchmark
    results = benchmark(
        checkpoint_dir=Path(args.checkpoint_dir),
        evaluators=eval_types
    )
    
    logger.info(f"Benchmark complete. Overall results: {results.get('overall', {})}")
