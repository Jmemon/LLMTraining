from pathlib import Path
import torch
import wandb
import logging
import time
import math
from typing import Optional, Dict, Any, List, Tuple, Union
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

from vorox.configs import RunConfig, EvaluatorType
from vorox.optimizer import OptimizerBase
from vorox.loss import LossBase
from vorox.evaluators.builder import EvaluatorBuilder
from vorox.metrics import MetricsCalculator
from vorox.vorox import Vorox
from vorox.data.builder import DatasetBuilder

logger = logging.getLogger(__name__)

def train(cfg: RunConfig) -> Dict[str, Any]:
    """
    Executes the complete training pipeline for Vorox language models with comprehensive monitoring and checkpointing.
    
    This function orchestrates the end-to-end training process, including model initialization,
    optimization, validation, early stopping, and checkpoint management. It implements a multi-dataset
    training strategy with configurable metrics tracking and hardware acceleration support.
    
    Architecture:
        - Implements a modular training loop with O(epochs * batches) time complexity
        - Supports mixed precision training with automatic gradient scaling
        - Maintains model state through configurable checkpointing strategy
        - Integrates with W&B for experiment tracking with O(log_interval) logging frequency
        - Implements early stopping with patience-based termination
    
    Parameters:
        cfg (RunConfig): Comprehensive configuration object containing nested configurations for:
            - experiment_name (str): Unique identifier for the training run
            - seed (int): Random seed for reproducibility
            - model (ModelConfig): Model architecture and parameters
            - optimizer (OptimizerConfig): Optimization algorithm and learning rate schedule
            - data (DataConfig): Dataset selection and dataloader parameters
            - loss (LossConfig): Loss function specification
            - hardware (HardwareConfig): Device and precision settings
            - logging (LoggingConfig): W&B integration parameters
            - checkpoint (CheckpointConfig): Model persistence strategy
            - metrics (MetricsConfig): Evaluation metrics and validation frequency
            - train (TrainConfig): Training loop parameters (epochs)
            - eval (EvaluatorConfig): Evaluation configuration
    
    Returns:
        Dict[str, Any]: Dictionary containing training results and metrics:
            - 'best_checkpoint_path': Path to the best checkpoint file
            - 'best_metric_value': Value of the best metric achieved
            - 'final_loss': Final training loss value
            - 'eval_results': Results from the final evaluation
    
    Raises:
        ValueError: If required configuration fields are missing or invalid
    """
    # Validate critical configuration fields
    if not cfg.experiment_name:
        raise ValueError("experiment_name must be specified in RunConfig")
    if not cfg.checkpoint.checkpoint_dir:
        raise ValueError("checkpoint_dir must be specified in CheckpointConfig")
    if cfg.hardware.device not in ["cpu", "cuda", "mps"]:
        raise ValueError(f"Unsupported device: {cfg.hardware.device}")
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger.info(f"Starting training for experiment: {cfg.experiment_name}")
    
    # Initialize W&B if enabled
    if cfg.logging.wandb_project:
        wandb.init(
            project=cfg.logging.wandb_project,
            entity=cfg.logging.wandb_entity,
            name=cfg.logging.wandb_run_name or cfg.experiment_name,
            tags=cfg.logging.wandb_tags,
            config=cfg.model_dump()
        )
        logger.info(f"Initialized W&B project: {cfg.logging.wandb_project}")
    
    # Set random seed for reproducibility
    torch.manual_seed(cfg.seed)
    if cfg.hardware.device == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)
    
    # Determine device
    device = torch.device(cfg.hardware.device)
    
    # Initialize tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.architecture.tokenizer_name)
    logger.info(f"Initialized tokenizer: {cfg.architecture.tokenizer_name}")
    
    # Initialize model
    model = Vorox(cfg, tokenizer.vocab_size).to(device)
    logger.info(f"Initialized Vorox model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Build datasets and dataloaders
    train_datasets = DatasetBuilder.build(cfg)
    logger.info(f"Built {len(train_datasets)} training datasets")
    
    # Create DataLoader
    train_loader = DataLoader(
        train_datasets[0],  # Assuming first dataset is the main one
        batch_size=cfg.data.micro_batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=True if cfg.hardware.device == "cuda" else False,
        shuffle=cfg.data.shuffle_buffer,
        prefetch_factor=cfg.data.prefetch_size if hasattr(cfg.data, 'prefetch_size') else 2,
    )
    logger.info(f"Created DataLoader with batch size {cfg.data.micro_batch_size}")
    
    # Initialize optimizer
    optimizer = OptimizerBase.build(model, cfg)
    logger.info(f"Initialized optimizer: {cfg.optimizer.type}")
    
    # Initialize loss function
    criterion = LossBase.build(cfg)
    logger.info(f"Initialized loss function: {cfg.loss.type}")
    
    # Initialize scheduler if specified
    scheduler = None
    if hasattr(cfg.optimizer, 'scheduler') and cfg.optimizer.scheduler:
        if cfg.optimizer.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg.train.epochs, eta_min=1e-6
            )
            logger.info("Initialized cosine learning rate scheduler")
        elif cfg.optimizer.scheduler == "constant":
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
            logger.info("Initialized constant learning rate scheduler")
    
    # Set up precision handling
    if cfg.hardware.precision == "fp16" and cfg.hardware.device == "cuda":
        scaler = torch.cuda.amp.GradScaler()
        use_amp = True
        logger.info("Using mixed precision training (fp16)")
    elif cfg.hardware.precision == "bf16" and cfg.hardware.device == "cuda" and torch.cuda.is_bf16_supported():
        # PyTorch 1.10+ supports native bf16 on supported hardware
        use_amp = True
        scaler = torch.cuda.amp.GradScaler()
        logger.info("Using mixed precision training (bf16)")
    else:
        use_amp = False
        scaler = None
        logger.info(f"Using full precision training ({cfg.hardware.precision})")
    
    # Create checkpoint directory
    checkpoint_dir = Path(cfg.checkpoint.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize tracking variables
    best_metric_value = float('inf') if cfg.checkpoint.mode == "min" else float('-inf')
    best_ckpt_path = None
    early_stop_counter = 0
    global_step = 0
    
    # Build evaluators if specified
    evaluators = EvaluatorBuilder.build(cfg.eval, cfg.eval.evaluators) if hasattr(cfg, 'eval') else []
    
    # Load from checkpoint if specified
    if cfg.checkpoint.load_from_checkpoint:
        checkpoint_path = Path(cfg.checkpoint.load_from_checkpoint)
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'global_step' in checkpoint:
                global_step = checkpoint['global_step']
            logger.info(f"Loaded checkpoint from {checkpoint_path}")
        else:
            logger.warning(f"Checkpoint file {checkpoint_path} not found, starting from scratch")
    
    # Training loop
    for epoch in range(cfg.train.epochs):
        logger.info(f"Starting epoch {epoch+1}/{cfg.train.epochs}")
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        # Progress bar for training
        train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.train.epochs}")
        
        for batch_idx, batch in enumerate(train_iterator):
            # Process batch - expecting dict with 'text' key containing list of strings
            text_samples = batch['text']
            
            # Tokenize the input text with padding/truncation
            tokenized_inputs = tokenizer(
                text_samples,
                padding="max_length",
                truncation=True,
                max_length=cfg.data.max_seq_len,
                return_tensors="pt"
            ).to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass with optional AMP
            if use_amp:
                with torch.cuda.amp.autocast():
                    # Pass tokenized inputs to model with causal attention mask
                    outputs = model(
                        tokenized_inputs.input_ids,
                        causal_attn_mask=True,
                        apply_softmax=True
                    )
                    
                    # Calculate loss using the criterion
                    # For autoregressive training, shift the inputs to create targets
                    input_ids = tokenized_inputs.input_ids
                    targets = input_ids[:, 1:].contiguous()  # Shift right to get targets
                    model_outputs = outputs[:, :-1, :].contiguous()  # Remove last token prediction
                    
                    loss = criterion(model_outputs.view(-1, tokenizer.vocab_size), targets.view(-1))
                
                # Backward pass with scaler
                scaler.scale(loss).backward()
                
                # Gradient clipping
                if cfg.optimizer.gradient_clip_val > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optimizer.gradient_clip_val)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard forward/backward pass
                outputs = model(
                    tokenized_inputs.input_ids,
                    causal_attn_mask=True,
                    apply_softmax=True
                )
                
                # Calculate loss using the criterion
                # For autoregressive training, shift the inputs to create targets
                input_ids = tokenized_inputs.input_ids
                targets = input_ids[:, 1:].contiguous()  # Shift right to get targets
                model_outputs = outputs[:, :-1, :].contiguous()  # Remove last token prediction
                
                loss = criterion(model_outputs.view(-1, tokenizer.vocab_size), targets.view(-1))
                loss.backward()
                
                # Gradient clipping
                if cfg.optimizer.gradient_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optimizer.gradient_clip_val)
                
                optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1
            
            # Update progress bar
            train_iterator.set_postfix({"loss": loss.item()})
            
            # Calculate training metrics if enabled
            train_metrics_log = {}
            if cfg.metrics.compute_metrics and global_step % cfg.logging.log_every_n_steps == 0:
                batch_metrics = MetricsCalculator.calculate_metrics(
                    model_outputs, targets, cfg.metrics.train_metrics
                )
                for metric_name, metric_value in batch_metrics.items():
                    train_metrics_log[f"train/{metric_name}"] = metric_value
            
            # Log metrics
            if global_step % cfg.logging.log_every_n_steps == 0:
                log_message = f"Step {global_step}, Loss: {loss.item():.6f}"
                
                # Add any calculated metrics to the log message
                for metric_name, metric_value in train_metrics_log.items():
                    metric_short_name = metric_name.split('/')[-1]
                    log_message += f", {metric_short_name}: {metric_value:.6f}"
                
                logger.info(log_message)
                
                if cfg.logging.wandb_project:
                    wandb_log = {
                        "train/loss": loss.item(),
                        "epoch": epoch,
                        "global_step": global_step,
                        "learning_rate": optimizer.param_groups[0]['lr'],
                        **train_metrics_log
                    }
                    wandb.log(wandb_log)
            
            # Run evaluation if needed
            if evaluators and hasattr(cfg.eval, 'val_check_interval') and global_step % cfg.eval.val_check_interval == 0:
                model.eval()
                eval_results = {}
                
                for evaluator in evaluators:
                    logger.info(f"Running evaluation with {evaluator.name}")
                    eval_result = evaluator(model)
                    eval_results[evaluator.name.value] = eval_result
                    
                    # Log evaluation results
                    if cfg.logging.wandb_project:
                        for metric_name, metric_value in eval_result.items():
                            wandb.log({f"eval/{evaluator.name.value}/{metric_name}": metric_value, "global_step": global_step})
                
                # Check if we should save a checkpoint based on monitored metric
                if cfg.checkpoint.monitor:
                    monitor_name = cfg.checkpoint.monitor.value
                    if monitor_name in eval_results:
                        # Get the main metric from the evaluator results
                        current_metric = eval_results[monitor_name].get("accuracy", 0.0)
                        
                        is_best = False
                        if cfg.checkpoint.mode == "min" and current_metric < best_metric_value:
                            best_metric_value = current_metric
                            is_best = True
                            early_stop_counter = 0
                        elif cfg.checkpoint.mode == "max" and current_metric > best_metric_value:
                            best_metric_value = current_metric
                            is_best = True
                            early_stop_counter = 0
                        else:
                            early_stop_counter += 1
                        
                        if is_best:
                            best_ckpt_path = checkpoint_dir / f"{cfg.experiment_name}_best.pt"
                            torch.save({
                                'epoch': epoch,
                                'global_step': global_step,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': loss.item(),
                                'metric': current_metric,
                                'config': cfg.model_dump(),
                            }, best_ckpt_path)
                            logger.info(f"Saved best model checkpoint to {best_ckpt_path} with {monitor_name}: {best_metric_value:.6f}")
                
                # Early stopping check
                if hasattr(cfg.metrics, 'early_stopping') and cfg.metrics.early_stopping and early_stop_counter >= cfg.metrics.early_stopping_patience:
                    logger.info(f"Early stopping triggered after {early_stop_counter} evaluations without improvement")
                    break
                
                # Return to training mode
                model.train()
            
            # Save checkpoint periodically
            if cfg.checkpoint.save_every_n_steps > 0 and global_step % cfg.checkpoint.save_every_n_steps == 0:
                step_ckpt_path = checkpoint_dir / f"{cfg.experiment_name}_step_{global_step}.pt"
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                    'config': cfg.model_dump(),
                }, step_ckpt_path)
                logger.info(f"Saved checkpoint at step {global_step} to {step_ckpt_path}")
            
            # Update learning rate scheduler if it exists
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # This type of scheduler needs the validation loss
                    pass  # Will be updated after validation
                else:
                    scheduler.step()
        
        # Calculate average epoch loss
        avg_epoch_loss = epoch_loss / num_batches
        epoch_time = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch+1} completed in {epoch_time:.2f}s, Avg Loss: {avg_epoch_loss:.6f}")
        
        if cfg.logging.wandb_project:
            wandb.log({"train/epoch_loss": avg_epoch_loss, "epoch": epoch})
        
        # Final evaluation at the end of each epoch
        if evaluators:
            model.eval()
            eval_results = {}
            
            for evaluator in evaluators:
                logger.info(f"Running end-of-epoch evaluation with {evaluator.name}")
                eval_result = evaluator(model)
                eval_results[evaluator.name.value] = eval_result
                
                # Log evaluation results
                if cfg.logging.wandb_project:
                    for metric_name, metric_value in eval_result.items():
                        wandb.log({f"eval/{evaluator.name.value}/{metric_name}": metric_value, "epoch": epoch})
            
            # Update learning rate scheduler if it's ReduceLROnPlateau
            if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # Use the monitored metric for scheduler if available
                if cfg.checkpoint.monitor:
                    monitor_name = cfg.checkpoint.monitor.value
                    if monitor_name in eval_results:
                        current_metric = eval_results[monitor_name].get("accuracy", 0.0)
                        # For ReduceLROnPlateau, lower is better, so invert if mode is "max"
                        scheduler_metric = -current_metric if cfg.checkpoint.mode == "max" else current_metric
                        scheduler.step(scheduler_metric)
                else:
                    # Fall back to validation loss
                    scheduler.step(avg_epoch_loss)
    
    # Save final model if requested
    if cfg.checkpoint.save_last:
        last_ckpt_path = checkpoint_dir / f"{cfg.experiment_name}_last.pt"
        torch.save({
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_epoch_loss,
            'config': cfg.model_dump(),
        }, last_ckpt_path)
        logger.info(f"Saved final model checkpoint to {last_ckpt_path}")
    
    # Close wandb run if it was used
    if cfg.logging.wandb_project:
        wandb.finish()
    
    # Return training results
    results = {
        'best_checkpoint_path': str(best_ckpt_path) if best_ckpt_path else None,
        'best_metric_value': best_metric_value,
        'final_loss': avg_epoch_loss,
        'eval_results': eval_results if evaluators else None,
    }
    
    return results
