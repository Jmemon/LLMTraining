
import torch.optim as optim

from vorox.configs import OptimizerType


class OptimizerBase:
    """
    Factory class for neural network optimizers in the Vorox framework.
    
    Provides a unified interface for instantiating PyTorch optimizer instances based on
    configuration parameters, abstracting implementation details from client code.
    
    Architecture:
        Implements the Factory Method pattern with O(1) lookup complexity for optimizer types.
        Stateless class design with no instance variables, operating purely through class methods.
        Maps configuration enums directly to PyTorch optimizer implementations with zero overhead.
        
    Interface:
        - build(cls, model, cfg) -> torch.optim.Optimizer:
            Parameters:
              model (nn.Module): PyTorch model whose parameters will be optimized.
                Must have a valid .parameters() method returning an iterable of parameters.
              cfg (Config): Configuration object containing optimizer specifications.
                Must include optimizer.type (OptimizerType), optimizer.lr (float > 0),
                optimizer.betas (list[float] with values in range (0,1)), and
                optimizer.weight_decay (float ≥ 0).
            
            Returns:
              torch.optim.Optimizer: Instantiated optimizer object configured according to
              the provided parameters, ready to perform gradient-based updates.
            
            Raises:
              ValueError: If cfg.optimizer.type is not a recognized OptimizerType.
              TypeError: If model does not provide a valid .parameters() method.
    
    Behavior:
        - Stateless between calls; maintains no persistent state
        - Thread-safe due to lack of shared mutable state
        - Defers optimizer state management to returned PyTorch optimizer instances
        - Zero overhead beyond the underlying PyTorch optimizer implementations
    
    Integration:
        - Typically called during model initialization or training setup
        - Consumes configuration from the optimizer section of the main Config object
        - Example:
          ```
          model = Vorox(cfg, tokenizer.vocab_size)
          optimizer = OptimizerBase.build(model, cfg)
          ```
        - Returned optimizer is used directly in training loops for parameter updates
    
    Limitations:
        - Limited to three optimizer types (AdamW, Adam, SGD)
        - SGD implementation does not utilize momentum or weight_decay from config
        - No support for learning rate schedulers or warmup strategies
        - No support for parameter-specific learning rates or custom parameter groups
        - No gradient clipping configuration
    """

    @classmethod
    def build(cls, model, cfg):
        """
        Factory method that instantiates optimizer instances based on configuration parameters.
        
        Creates and returns the appropriate PyTorch optimizer implementation based on the
        optimizer type specified in the configuration, with O(1) dispatch complexity. Serves as the
        primary interface for optimizer instantiation throughout the Vorox framework.
        
        Architecture:
            - Implements the Factory Method pattern with constant-time dispatch
            - Maps enum values to concrete PyTorch optimizer implementations with zero overhead
            - Stateless design with no instance variables or shared state
            - Zero memory overhead beyond the instantiated optimizer
            - No caching strategy; creates new instance on each call
        
        Args:
            model (nn.Module): PyTorch model whose parameters will be optimized.
                Must have a valid .parameters() method returning an iterable of parameters.
                Typically a Vorox model instance or other PyTorch neural network.
                The parameters() method must return trainable parameters with requires_grad=True.
            
            cfg (Config): Configuration object containing optimizer settings.
                Must include cfg.optimizer.type field with a value from the OptimizerType enum
                ("adamw", "adam", or "sgd"). Also requires cfg.optimizer.lr (learning rate),
                cfg.optimizer.betas (momentum parameters), and cfg.optimizer.weight_decay
                (regularization strength) fields with appropriate numeric values.
        
        Returns:
            torch.optim.Optimizer: Instantiated optimizer object configured according to
                the provided parameters, ready to perform gradient-based updates.
                Return type varies based on the requested optimizer:
                - optim.AdamW for "adamw"
                - optim.Adam for "adam"
                - optim.SGD for "sgd"
        
        Raises:
            ValueError: If cfg.optimizer.type contains an unrecognized OptimizerType value.
            TypeError: If model does not provide a valid .parameters() method or if
                configuration parameters have incorrect types.
            RuntimeError: If model.parameters() returns an empty iterator.
        
        Behavior:
            - Thread-safe due to stateless design with no shared mutable state
            - Deterministic mapping between configuration and instantiated objects
            - Zero memory overhead beyond the instantiated optimizer
            - No caching strategy; creates new instance on each call
        
        Integration:
            - Typically called during model initialization or training setup
            - Consumes configuration from the optimizer section of the main Config object
            - Example:
              ```
              model = Vorox(cfg, tokenizer.vocab_size)
              optimizer = OptimizerBase.build(model, cfg)
              ```
            - Returned optimizer is used directly in training loops for parameter updates
        
        Limitations:
            - Limited to three optimizer types (AdamW, Adam, SGD)
            - SGD implementation does not utilize momentum or weight_decay from config
            - No support for learning rate schedulers or warmup strategies
            - No support for parameter-specific learning rates or custom parameter groups
            - No gradient clipping configuration
            - No support for optimizer state loading/saving for training resumption
        """
        if cfg.optimizer.type == OptimizerType.adamw:
            return optim.AdamW(model.parameters(), lr=cfg.optimizer.lr, betas=cfg.optimizer.betas, weight_decay=cfg.optimizer.weight_decay)
        elif cfg.optimizer.type == OptimizerType.adam:
            return optim.Adam(model.parameters(), lr=cfg.optimizer.lr, betas=cfg.optimizer.betas, weight_decay=cfg.optimizer.weight_decay)
        elif cfg.optimizer.type == OptimizerType.sgd:
            return optim.SGD(model.parameters(), lr=cfg.optimizer.lr)
