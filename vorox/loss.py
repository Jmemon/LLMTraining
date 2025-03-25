
import torch.nn as nn

from vorox.configs import LossType


class LossBase:
    """
    Factory class for neural network loss functions in the Vorox framework.
    
    Provides a unified interface for instantiating loss function modules based on
    configuration parameters, abstracting implementation details from client code.
    
    Architecture:
        Implements the Factory Method pattern with O(1) lookup complexity for loss types.
        Stateless class design with no instance variables, operating purely through class methods.
        Maps configuration enums directly to PyTorch loss implementations with zero overhead.
        
    Interface:
        - build(cls, cfg: Config) -> nn.Module:
            Factory method that instantiates appropriate loss function based on configuration.
            Parameters:
                cfg: Config object containing loss.type field (must be valid LossType enum)
            Returns:
                Instantiated nn.Module loss function ready for use in training
            Raises:
                No explicit exceptions, but will propagate any from nn.CrossEntropyLoss
                
    Behavior:
        - Thread-safe due to stateless design with no shared mutable state
        - Deterministic mapping between configuration and instantiated objects
        - Zero memory overhead beyond the instantiated loss function
        - No caching strategy; creates new instance on each call
        
    Integration:
        - Initialized via class method without instantiation: loss_fn = LossBase.build(cfg)
        - Directly consumed by training loop for computing loss values
        - Example:
          ```
          cfg = Config.model_validate(yaml.safe_load(config_file))
          loss_fn = LossBase.build(cfg)
          loss = loss_fn(model_output, target)
          ```
        
    Limitations:
        - Limited to predefined loss functions; custom losses require code changes
        - No parameterization of loss functions beyond their default configurations
        - No support for loss function composition or weighted combinations
        - Perplexity is implemented as standard cross-entropy without exponential transformation
        - No explicit handling for numerical stability issues in loss computation
    """
    @classmethod
    def build(cls, cfg):
        """
        Factory method that instantiates loss function modules based on configuration parameters.
        
        Creates and returns the appropriate PyTorch loss function implementation based on the
        loss type specified in the configuration, with O(1) dispatch complexity. Serves as the
        primary interface for loss function instantiation throughout the Vorox framework.
        
        Architecture:
            - Implements the Factory Method pattern with constant-time dispatch
            - Maps enum values to concrete PyTorch loss implementations with zero overhead
            - Stateless design with no instance variables or shared state
            - Zero memory overhead beyond the instantiated loss function
            - No caching strategy; creates new instance on each call
        
        Args:
            cfg (Config): Configuration object containing loss settings.
                Must include cfg.loss.type field with a value from the LossType enum
                (currently only "perplexity").
        
        Returns:
            nn.Module: Instantiated loss function module ready for integration into
                training loops. Currently always returns nn.CrossEntropyLoss.
        
        Behavior:
            - Thread-safe due to stateless design with no shared mutable state
            - Deterministic mapping between configuration and instantiated objects
            - Zero memory overhead beyond the instantiated loss function
            - No caching strategy; creates new instance on each call
        
        Integration:
            - Initialized via class method without instantiation: loss_fn = LossBase.build(cfg)
            - Directly consumed by training loop for computing loss values
            - Example:
              ```
              cfg = Config.model_validate(yaml.safe_load(config_file))
              loss_fn = LossBase.build(cfg)
              loss = loss_fn(model_output, target)
              ```
        
        Limitations:
            - Limited to predefined loss functions; custom losses require code changes
            - No parameterization of loss functions beyond their default configurations
            - Perplexity is implemented as standard cross-entropy without exponential transformation
            - No explicit handling for numerical stability issues in loss computation
            - No support for loss function composition or weighted combinations
        """
        if cfg.loss.type == LossType.perplexity:
            return nn.CrossEntropyLoss()
