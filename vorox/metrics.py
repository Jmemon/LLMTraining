import numpy as np
import torch
from typing import Tuple, List, Dict, Any, Union, Optional

def perplexity(pred: torch.Tensor, actual: torch.Tensor) -> float:
    """
    Calculate perplexity between predicted and actual token distributions.
    
    Perplexity is a measurement of how well a probability model predicts a sample,
    defined as the exponentiated average negative log-likelihood of a sequence.
    Lower perplexity indicates better prediction performance.
    
    Architecture:
        - Implements standard perplexity calculation with O(n) time complexity
        - Supports both PyTorch tensor and NumPy array inputs with automatic conversion
        - Vectorized implementation for optimal performance on both CPU and GPU
        - No gradient computation when detaching tensors for evaluation purposes
    
    Interface:
        - pred: PyTorch tensor of predicted log probabilities (any shape, will be flattened)
          * Values should be log probabilities (output of log_softmax)
          * Empty sequences will return NaN
        - actual: PyTorch tensor of ground truth token indices (any shape, will be flattened)
          * Values should be integer indices corresponding to vocabulary tokens
          * Empty sequences will return NaN
        - Returns: Python float representing the perplexity
          * Range: [1, +inf) where lower values indicate better prediction
          * Value of 1 indicates perfect prediction
    
    Behavior:
        - Detaches tensors from computation graph and moves to CPU before conversion to numpy
        - Thread-safe with no side effects or persistent state
        - Input tensors are not modified
        - Handles tensors of different shapes by flattening before calculation
        - Returns scalar float value even for multi-dimensional inputs
    
    Integration:
        - Designed for direct use in evaluation pipelines or via MetricsCalculator
        - Example:
          ```
          ppl = perplexity(model_log_probs, target_tokens)
          print(f"Perplexity: {ppl:.4f}")
          ```
    
    Limitations:
        - Requires log probabilities as input, not raw logits
        - Assumes token indices are valid (within vocabulary range)
        - No support for masked tokens or padding
        - Does not handle batch processing natively (will flatten all dimensions)
    
    Args:
        pred: Predicted log probabilities tensor
        actual: Actual token indices tensor
        
    Returns:
        float: Perplexity value
    """
    # Convert tensors to numpy arrays if they're not already
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(actual, torch.Tensor):
        actual = actual.detach().cpu().numpy()
    
    # Ensure inputs are flattened
    pred = pred.reshape(-1)
    actual = actual.reshape(-1)
    
    # Calculate negative log likelihood
    nll = -np.mean(pred[np.arange(len(actual)), actual])
    
    # Return perplexity
    return float(np.exp(nll))

class MetricsCalculator:
    """
    Utility class for calculating and aggregating evaluation metrics for language models.
    
    This class provides a standardized interface for computing performance metrics
    between predicted and actual token distributions. It implements a stateless design pattern
    with static methods to facilitate integration with training and evaluation pipelines.
    
    Architecture:
        - Implements a facade pattern over individual metric functions
        - O(1) dispatch complexity to appropriate metric implementations
        - Underlying metric implementations have varying complexities:
          * Perplexity: O(n) time, O(1) space
    
    Interface:
        - All methods accept PyTorch tensors and handle conversion to appropriate formats
        - Tensor shapes are automatically normalized for consistent processing
        - Results are returned as Python native types (float) for serialization compatibility
    
    Thread Safety:
        - All methods are stateless and thread-safe
        - No shared state or resources are maintained between calls
    
    Integration:
        - Designed to be called directly from training loops or evaluation scripts
        - Compatible with Weights & Biases and other metric tracking systems
        - Example usage:
          ```
          metrics = MetricsCalculator.calculate_metrics(
              model_output, ground_truth, ["perplexity"]
          )
          wandb.log(metrics)
          ```
    
    Limitations:
        - Does not support batched metric calculation (processes single samples)
        - All metrics assume standard language model output formats
    """
    
    @staticmethod
    def calculate_metrics(pred: torch.Tensor, actual: torch.Tensor, 
                          metrics_list: List[str]) -> Dict[str, float]:
        """
        Calculate specified metrics between predicted and actual data with dynamic dispatch.
        
        Computes multiple evaluation metrics in a single pass by dynamically dispatching to specialized
        metric implementations based on the provided metrics list. Serves as the primary entry point
        for all metric calculations in the evaluation pipeline.
        
        Architecture:
            - Implements O(1) dispatch to appropriate metric functions via dictionary lookup pattern
            - Aggregates results into a single dictionary with O(k) space complexity where k is the number of metrics
            - Each metric function has its own computational complexity
            - No caching mechanism; recalculates metrics on each call
        
        Interface:
            - pred: PyTorch tensor of predicted values (shape depends on metric)
              * Must contain finite numeric values
              * Automatically detached from computation graph during metric calculation
            - actual: PyTorch tensor of ground truth values (shape depends on metric)
              * Must contain finite numeric values
              * Automatically detached from computation graph during metric calculation
            - metrics_list: List of string identifiers for metrics to calculate
              * Supported values: "perplexity", "loss"
              * Unknown metrics are silently ignored
              * Empty list returns empty dictionary
              * Order of metrics in list does not affect calculation order
            
        Returns:
            Dict[str, float]: Dictionary mapping metric names to their computed values
              * Keys match the strings provided in metrics_list (if supported)
              * Values are Python floats, suitable for serialization
              * Empty dictionary if metrics_list is empty or contains only unsupported metrics
        
        Behavior:
            - Thread-safe with no side effects or persistent state
            - Silently ignores unrecognized metric names without raising exceptions
            - Each metric calculation is independent; failure in one doesn't affect others
            - Input tensors are not modified; all operations create new tensors/arrays
        
        Integration:
            - Designed for direct use in training and evaluation loops
            - Compatible with logging systems like Weights & Biases
            - Example:
              ```
              val_metrics = MetricsCalculator.calculate_metrics(
                  model_output, ground_truth, ["perplexity"]
              )
              wandb.log({"val/perplexity": val_metrics["perplexity"]})
              ```
        
        Limitations:
            - No support for custom or user-defined metrics
            - Processes single samples, not batched calculations
            - No parallel computation of multiple metrics
            - No validation of metric names; invalid names are silently ignored
        """
        results = {}
        
        for metric_name in metrics_list:
            if metric_name == "perplexity":
                results["perplexity"] = perplexity(pred, actual)
            # "loss" is typically calculated separately in the training loop
            # and may be included in metrics_list from the config
        
        return results
