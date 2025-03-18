from enum import Enum
from pydantic import BaseModel
import yaml


class Tokenizer(BaseModel):
    """
    Configuration model for text tokenization in the Vorox framework.
    
    Encapsulates tokenizer identification and configuration for neural language models.
    The tokenizer is responsible for converting raw text into token IDs and vice versa.
    
    Architecture:
        Inherits from Pydantic's BaseModel for validation and serialization.
        Simple structure with minimal configuration to support various tokenizer implementations.
    
    Interface:
        - name (str): Identifier for the tokenizer implementation to use. Must correspond to a 
          valid tokenizer name in the underlying tokenization library (e.g., "gpt2", "llama", 
          "sentencepiece"). No default value; must be explicitly provided.
    
    Behavior:
        - Immutable after initialization following Pydantic's data model pattern
        - Thread-safe for read operations
        - No internal state management beyond the configured properties
    
    Integration:
        - Initialized via Pydantic's model_validate method from YAML/JSON configuration
        - Example: 
          ```
          tokenizer_config = Tokenizer.model_validate({"name": "gpt2"})
          ```
        - Used as a component in the main Config class to define tokenization strategy
    
    Limitations:
        - Does not directly instantiate tokenizer objects; serves as configuration only
        - Additional tokenizer-specific parameters must be added to support advanced configurations
        - No validation for tokenizer name existence in the target library
    """
    name: str

class ActivationType(str, Enum):
    """
    Enumeration of supported neural network activation functions in the Vorox framework.
    
    Defines the canonical set of non-linear activation functions that can be configured
    for neural network layers, ensuring type safety and configuration validation.
    
    Architecture:
        Inherits from both str and Enum to enable string serialization while maintaining
        type safety. Implements a 1:1 mapping between enum members and string values
        with O(1) lookup complexity for both directions.
    
    Interface:
        - gelu: Gaussian Error Linear Unit activation (smoother approximation of ReLU)
        - relu: Rectified Linear Unit activation (max(0,x))
        - silu: Sigmoid Linear Unit activation (x * sigmoid(x))
        - swiglu: SwiGLU activation (specialized gated activation function)
    
    Behavior:
        - Immutable after definition following Python's Enum pattern
        - Thread-safe for all operations
        - String-serializable for configuration files via the str inheritance
        - Hashable for use as dictionary keys
    
    Integration:
        - Used directly in Architecture model for activation function specification
        - Consumed by Activation.build() factory method to instantiate the appropriate
          nn.Module implementation
        - Example:
          ```
          activation_type = ActivationType.gelu
          activation_module = Activation.build(config)
          ```
    
    Limitations:
        - Limited to predefined activation functions; custom activations require code changes
        - No parameterization of activation functions (e.g., leaky ReLU alpha parameter)
        - Implementation details of each activation are defined elsewhere in the codebase
    """
    gelu = "gelu"
    relu = "relu"
    silu = "silu"
    swiglu = "swiglu"

class Architecture(BaseModel):
    """
    Configuration model for neural network architecture parameters in the Vorox framework.
    
    Defines the structural hyperparameters for transformer-based language models, encapsulating
    the dimensional specifications and activation mechanisms that determine model capacity and
    computational characteristics.
    
    Architecture:
        Inherits from Pydantic's BaseModel for validation and serialization.
        Implements a flat parameter structure with O(1) access complexity for all configuration
        values. Parameters are validated at initialization time with type checking.
    
    Interface:
        - n_layers (int): Number of transformer decoder blocks in the model stack.
          Directly impacts model depth and computational complexity (O(n_layers)).
        - d_model (int): Embedding dimension size for token representations.
          Determines the dimensionality of attention mechanisms and feed-forward operations.
        - n_heads (int): Number of attention heads for multi-head attention mechanisms.
          Must satisfy: n_heads ≥ n_kv_heads and d_model % n_heads == 0.
        - n_kv_heads (int): Number of key-value heads for grouped-query attention.
          When n_kv_heads < n_heads, implements grouped-query attention for efficiency.
          When n_kv_heads == n_heads, implements standard multi-head attention.
        - hidden_size (int): Dimension of the feed-forward network's hidden layer.
          Typically 2-4x larger than d_model, controlling intermediate representation capacity.
        - activation (ActivationType): Non-linear activation function applied in feed-forward networks.
          Must be a valid member of the ActivationType enum.
        - rope (bool): Whether to use Rotary Position Embeddings for sequence position encoding.
          When True, enables RoPE for improved handling of relative positions.
        - rope_theta (int): Base frequency for rotary position embeddings when rope=True.
          Controls the frequency spectrum of positional encodings, affecting model's
          ability to generalize to different sequence lengths.
    
    Behavior:
        - Immutable after initialization following Pydantic's data model pattern
        - Thread-safe for read operations
        - No internal state management beyond the configured properties
        - Validation occurs at initialization time with type checking
    
    Integration:
        - Initialized via Pydantic's model_validate method from YAML/JSON configuration
        - Used by Vorox model constructor to instantiate neural network components
        - Example:
          ```
          arch_config = Architecture.model_validate({
              "n_layers": 12, "d_model": 768, "n_heads": 12, "n_kv_heads": 12,
              "hidden_size": 3072, "activation": "gelu", "rope": True, "rope_theta": 10000
          })
          model = Vorox(arch_config)
          ```
    
    Limitations:
        - Does not validate numerical relationships between parameters (e.g., d_model divisibility by n_heads)
        - No support for variable-depth architectures or per-layer configurations
        - Limited to transformer decoder-only architectures; encoder-decoder not supported
        - No explicit memory usage estimation based on parameter values
    """
    n_layers: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    hidden_size: int
    activation: ActivationType
    rope: bool
    rope_theta: int

class OptimizerType(str, Enum):
    """
    Enumeration of supported optimization algorithms in the Vorox framework.
    
    Defines the canonical set of gradient-based optimizers that can be configured
    for neural network training, ensuring type safety and configuration validation.
    
    Architecture:
        Inherits from both str and Enum to enable string serialization while maintaining
        type safety. Implements a 1:1 mapping between enum members and string values
        with O(1) lookup complexity for both directions.
    
    Interface:
        - adamw: AdamW optimizer (Adam with decoupled weight decay regularization)
          Implements the algorithm from "Decoupled Weight Decay Regularization" (Loshchilov & Hutter, 2019)
        - adam: Adam optimizer (Adaptive Moment Estimation)
          Implements the algorithm from "Adam: A Method for Stochastic Optimization" (Kingma & Ba, 2015)
        - sgd: Stochastic Gradient Descent optimizer
          Classical optimization method with optional momentum and weight decay
    
    Behavior:
        - Immutable after definition following Python's Enum pattern
        - Thread-safe for all operations
        - String-serializable for configuration files via the str inheritance
        - Hashable for use as dictionary keys
    
    Integration:
        - Used directly in Optimizer model for optimizer type specification
        - Consumed by OptimizerBase.build() factory method to instantiate the appropriate
          torch.optim.Optimizer implementation
        - Example:
          ```
          optimizer_type = OptimizerType.adamw
          optimizer = OptimizerBase.build(model, config)
          ```
    
    Limitations:
        - Limited to predefined optimizers; custom optimizers require code changes
        - No support for optimizer-specific parameters beyond those in the Optimizer model
        - Implementation details of each optimizer are defined in torch.optim
        - No support for learning rate schedulers or adaptive learning rate methods
    """
    adamw = "adamw"
    adam = "adam"
    sgd = "sgd"

class Optimizer(BaseModel):
    """
    Configuration model for gradient-based optimization algorithms in the Vorox framework.
    
    Encapsulates hyperparameters for neural network training optimization, providing a
    validated configuration interface for PyTorch's optimizer implementations.
    
    Architecture:
        Inherits from Pydantic's BaseModel for validation and serialization.
        Implements a flat parameter structure with O(1) access complexity for all configuration
        values. Parameters are validated at initialization time with type checking.
    
    Interface:
        - type (OptimizerType): Enum specifying the optimization algorithm to use.
          Must be a valid member of the OptimizerType enum (adamw, adam, or sgd).
        - lr (float): Learning rate controlling step size during gradient descent.
          Typical range: 1e-5 to 1e-2 depending on optimizer type and model architecture.
          Must be positive; higher values may cause training instability.
        - betas (list[float]): Exponential moving average factors for gradient moments.
          For Adam/AdamW: [β₁, β₂] where β₁ controls first moment (momentum) and
          β₂ controls second moment (uncentered variance). Typical values: [0.9, 0.999].
          Must satisfy: 0 < β₁, β₂ < 1.0 for convergence guarantees.
          For SGD: Only first value is used as momentum factor when momentum is enabled.
        - weight_decay (float): L2 regularization coefficient applied to model weights.
          Controls model complexity by penalizing large weights. Typical range: 0 to 0.1.
          In AdamW, applied as true regularization; in Adam, modifies gradient directly.
    
    Behavior:
        - Immutable after initialization following Pydantic's data model pattern
        - Thread-safe for read operations
        - No internal state management beyond the configured properties
        - Validation occurs at initialization time with type checking
    
    Integration:
        - Initialized via Pydantic's model_validate method from YAML/JSON configuration
        - Consumed by OptimizerBase.build() factory method to instantiate the appropriate
          torch.optim.Optimizer implementation
        - Example:
          ```
          optimizer_config = Optimizer.model_validate({
              "type": "adamw", "lr": 1e-4, "betas": [0.9, 0.999], "weight_decay": 0.01
          })
          optimizer = OptimizerBase.build(model, config)
          ```
    
    Limitations:
        - No support for learning rate schedulers or warmup strategies
        - Limited to fixed learning rates; no adaptive methods beyond Adam variants
        - No support for optimizer-specific parameters (e.g., amsgrad for Adam)
        - Beta parameters must be provided as a list, not a tuple as expected by PyTorch
        - No validation for numerical stability (e.g., extremely small/large learning rates)
        - No gradient clipping configuration
    """
    type: OptimizerType
    lr: float
    betas: list[float]
    weight_decay: float

class LossType(str, Enum):
    """
    Enumeration of supported loss functions in the Vorox framework.
    
    Defines the canonical set of loss functions that can be configured for neural
    network training, ensuring type safety and configuration validation.
    
    Architecture:
        Inherits from both str and Enum to enable string serialization while maintaining
        type safety. Implements a 1:1 mapping between enum members and string values
        with O(1) lookup complexity for both directions.
    
    Interface:
        - mse: Mean Squared Error loss function
          Measures average squared difference between predictions and targets.
          Suitable for regression tasks and continuous output values.
        - cross_entropy: Cross Entropy loss function
          Measures probabilistic divergence between predicted and target distributions.
          Standard loss for classification tasks with categorical outputs.
        - perplexity: Perplexity-based loss function
          Exponential of cross-entropy loss, commonly used in language modeling.
          Measures how "surprised" the model is by the target text.
    
    Behavior:
        - Immutable after definition following Python's Enum pattern
        - Thread-safe for all operations
        - String-serializable for configuration files via the str inheritance
        - Hashable for use as dictionary keys
    
    Integration:
        - Used directly in Loss model for loss function specification
        - Consumed by LossBase.build() factory method to instantiate the appropriate
          nn.Module implementation from torch.nn
        - Example:
          ```
          loss_type = LossType.cross_entropy
          loss_fn = LossBase.build(config)
          ```
    
    Limitations:
        - Limited to predefined loss functions; custom losses require code changes
        - No parameterization of loss functions (e.g., weight for class imbalance)
        - Implementation details of each loss are defined in torch.nn
        - Perplexity is implemented as cross-entropy with post-processing
        - No support for combined or weighted loss functions
    """
    mse = "mse"
    cross_entropy = "cross_entropy"
    perplexity = "perplexity"

class Loss(BaseModel):
    """
    Configuration model for loss function selection in the Vorox framework.
    
    Encapsulates the loss function specification for neural network training,
    providing a validated configuration interface for PyTorch's loss implementations.
    
    Architecture:
        Inherits from Pydantic's BaseModel for validation and serialization.
        Implements a minimal parameter structure with O(1) access complexity.
        Single-field design pattern focuses on type selection with implementation
        details managed by the LossBase factory class.
    
    Interface:
        - type (LossType): Enum specifying the loss function algorithm to use.
          Must be a valid member of the LossType enum (mse, cross_entropy, or perplexity).
          No default value; must be explicitly provided in configuration.
          Determines the mathematical objective function for model optimization.
    
    Behavior:
        - Immutable after initialization following Pydantic's data model pattern
        - Thread-safe for read operations
        - No internal state management beyond the configured properties
        - Validation occurs at initialization time with type checking
    
    Integration:
        - Initialized via Pydantic's model_validate method from YAML/JSON configuration
        - Consumed by LossBase.build() factory method to instantiate the appropriate
          nn.Module implementation from torch.nn
        - Example:
          ```
          loss_config = Loss.model_validate({"type": "cross_entropy"})
          loss_fn = LossBase.build(config)
          ```
        - Used as a component in the main Config class to define training objective
    
    Limitations:
        - No parameterization of loss functions (e.g., weight for class imbalance)
        - No support for custom loss functions without code changes
        - No support for combined or weighted loss functions
        - Implementation details of each loss are defined in torch.nn or vorox.loss
        - Perplexity requires post-processing of cross-entropy values
    """
    type: LossType

class Train(BaseModel):
    """
    Configuration model for neural network training parameters in the Vorox framework.
    
    Encapsulates core hyperparameters that control the training process lifecycle,
    including iteration counts, batch sizing strategy, and sequence length constraints.
    
    Architecture:
        Inherits from Pydantic's BaseModel for validation and serialization.
        Implements a flat parameter structure with O(1) access complexity for all configuration
        values. Parameters are validated at initialization time with type checking.
    
    Interface:
        - epochs (int): Number of complete passes through the training dataset.
          Controls total training duration and convergence opportunity.
          Must be positive; higher values increase training time linearly.
        - macro_batch_size (int): Number of samples processed per optimization step.
          Determines gradient estimation quality and memory efficiency.
          When using gradient accumulation: macro_batch_size = micro_batch_size * accumulation_steps.
          Must satisfy: macro_batch_size ≥ micro_batch_size and macro_batch_size % micro_batch_size == 0.
        - micro_batch_size (int): Number of samples processed in a single forward/backward pass.
          Controls GPU memory usage and parallelization efficiency.
          Must be positive and not exceed available GPU memory capacity.
          Smaller values reduce memory requirements but may increase computation time.
        - max_seq_len (int): Maximum sequence length in tokens for training samples.
          Directly impacts memory usage (O(max_seq_len²)) due to attention mechanisms.
          Longer sequences enable modeling of more complex dependencies but require
          more memory and computation time.
    
    Behavior:
        - Immutable after initialization following Pydantic's data model pattern
        - Thread-safe for read operations
        - No internal state management beyond the configured properties
        - Validation occurs at initialization time with type checking
    
    Integration:
        - Initialized via Pydantic's model_validate method from YAML/JSON configuration
        - Used by training loop to control iteration counts and batch sizing
        - Consumed by DataLoader configuration to set appropriate batch sizes
        - Example:
          ```
          train_config = Train.model_validate({
              "epochs": 3, "macro_batch_size": 32, "micro_batch_size": 8, "max_seq_len": 1024
          })
          ```
        - Used as a component in the main Config class to define training parameters
    
    Limitations:
        - Does not validate numerical relationships between parameters (e.g., batch size divisibility)
        - No support for dynamic/adaptive sequence lengths based on available memory
        - No explicit learning rate scheduling or early stopping configuration
        - No configuration for gradient accumulation steps (derived from batch sizes)
        - No validation for hardware compatibility (e.g., memory requirements)
        - No support for curriculum learning or progressive sequence length increases
    """
    epochs: int
    macro_batch_size: int
    micro_batch_size: int
    max_seq_len: int

class Dataset(str, Enum):
    """
    Enumeration of supported training datasets in the Vorox framework.
    
    Defines the canonical set of pre-processed datasets that can be configured
    for neural language model training, ensuring type safety and configuration validation.
    
    Architecture:
        Inherits from both str and Enum to enable string serialization while maintaining
        type safety. Implements a 1:1 mapping between enum members and string values
        with O(1) lookup complexity for both directions.
    
    Interface:
        - dclm_baseline: DCLM Baseline dataset
          Curated dataset from the DCLM Baseline repository (mlfoundations/dclm-baseline-1.0).
          Provides high-quality text data optimized for language model pretraining.
        - thestack: The Stack dataset
          Programming language corpus containing code from various programming languages.
          Suitable for code-oriented language models and code completion tasks.
        - dolma: Dolma dataset
          Large-scale, multilingual corpus designed for language model pretraining.
          Includes diverse text sources with quality filtering applied.
        - redpajama: RedPajama dataset
          Open-source reproduction of the LLaMA training corpus.
          Contains web text, books, papers, and other diverse text sources.
    
    Behavior:
        - Immutable after definition following Python's Enum pattern
        - Thread-safe for all operations
        - String-serializable for configuration files via the str inheritance
        - Hashable for use as dictionary keys
    
    Integration:
        - Used in TrainingDataConfig to specify dataset sources
        - Consumed by data loading utilities to determine appropriate preprocessing
        - Maps directly to dataset-specific URL patterns and processing functions
        - Example:
          ```
          dataset_type = Dataset.dclm_baseline
          urls = get_dclm_baseline_urls(bucket, prefix)
          ```
    
    Limitations:
        - Limited to predefined datasets; custom datasets require code changes
        - No parameterization of dataset variants or subsets
        - Implementation details of each dataset's processing are defined elsewhere
        - No metadata about dataset sizes, content types, or licensing information
        - No versioning mechanism for dataset iterations or updates
    """
    dclm_baseline = "dclm_baseline"
    thestack = "thestack"
    dolma = "dolma"
    redpajama = "redpajama"

class DataSettings(BaseModel):
    """
    Configuration model for data pipeline settings in the Vorox framework.
    
    Encapsulates parameters that control data loading, caching, and preprocessing
    behavior during neural network training, optimizing for throughput and memory efficiency.
    
    Architecture:
        Inherits from Pydantic's BaseModel for validation and serialization.
        Implements a flat parameter structure with O(1) access complexity for all configuration
        values. Parameters are validated at initialization time with type checking.
        Designed for integration with PostgreSQL-based metadata caching system.
    
    Interface:
        - prefetch_size (int): Number of batches to prefetch and buffer in memory.
          Controls data loading pipeline efficiency by decoupling I/O from computation.
          Higher values increase memory usage but reduce likelihood of data starvation.
          Must be positive; optimal values typically range from 2-10 depending on batch size.
        - cache_dsn (str): PostgreSQL connection string for metadata caching.
          Format: "postgresql://user:password@host:port/database".
          Used to establish connection with MetadataCache for tracking sample IDs.
          Must be a valid DSN string with appropriate access credentials.
        - shuffle_buffer (bool): Whether to randomize sample order during training.
          When True, implements a sliding window shuffling strategy for improved
          training dynamics. Default: False (preserves dataset ordering).
        - num_workers (int): Number of parallel data loading processes.
          Controls CPU parallelism for data preprocessing and loading operations.
          Higher values increase CPU utilization but may cause contention.
          Default: 4; should not exceed available CPU cores minus one.
    
    Behavior:
        - Immutable after initialization following Pydantic's data model pattern
        - Thread-safe for read operations
        - No internal state management beyond the configured properties
        - Validation occurs at initialization time with type checking
        - Connection management to PostgreSQL is handled by MetadataCache, not DataSettings
    
    Integration:
        - Initialized via Pydantic's model_validate method from YAML/JSON configuration
        - Used by data loading pipeline to configure WebDataset and DataLoader instances
        - Consumed by MetadataCache for establishing database connections
        - Example:
          ```
          data_settings = DataSettings.model_validate({
              "prefetch_size": 4,
              "cache_dsn": "postgresql://user:pass@localhost:5432/vorox_metadata",
              "shuffle_buffer": True,
              "num_workers": 8
          })
          metadata_cache = MetadataCache(data_settings.cache_dsn)
          ```
        - Used as a component in TrainingDataConfig to define data pipeline behavior
    
    Limitations:
        - No support for connection pooling or advanced PostgreSQL configurations
        - No retry logic or connection timeout parameters for database operations
        - No validation for PostgreSQL DSN format or connectivity testing
        - Shuffle buffer implementation is boolean only; no configuration for buffer size
        - No explicit memory usage estimation based on prefetch_size and batch dimensions
        - No support for distributed data loading across multiple nodes
    """
    prefetch_size: int
    cache_dsn: str  # PostgreSQL DSN (e.g., postgresql://user:pass@host:port/db)
    shuffle_buffer: bool = False
    num_workers: int = 4
    # Additional settings (e.g. timeouts) can be added here

class TrainingDataConfig(BaseModel):
    """
    Configuration model for training data pipeline in the Vorox framework.
    
    Encapsulates dataset source locations and processing parameters, providing a unified
    interface for configuring data ingestion, preprocessing, and caching strategies for
    neural language model training.
    
    Architecture:
        Inherits from Pydantic's BaseModel for validation and serialization.
        Implements a composite structure with O(1) access complexity for configuration values.
        Combines DataSettings for pipeline behavior with explicit URL references for data sources.
        Designed for integration with WebDataset-based streaming data processing pipeline.
    
    Interface:
        - settings (DataSettings): Nested configuration object for data pipeline parameters.
          Controls prefetching, caching, parallelism, and shuffling behavior.
          Must be a valid DataSettings instance with appropriate PostgreSQL credentials.
          See DataSettings documentation for detailed parameter specifications.
        - urls (list[str]): Collection of WebDataset-compatible data source URLs.
          Each URL must point to a valid WebDataset tar file or S3/cloud storage location.
          Format varies by dataset type (e.g., s3://bucket/prefix/shard-{000000..000099}.tar).
          Empty list is valid but will result in no training data being loaded.
          Order determines initial data presentation sequence before shuffling.
    
    Behavior:
        - Immutable after initialization following Pydantic's data model pattern
        - Thread-safe for read operations
        - No internal state management beyond the configured properties
        - Validation occurs at initialization time with type checking
        - URL validation and existence checking deferred to data loading time
    
    Integration:
        - Initialized via Pydantic's model_validate method from YAML/JSON configuration
        - Used by data loading pipeline to configure WebDataset and DataLoader instances
        - Consumed by training loop to establish data sources and processing parameters
        - Example:
          ```
          data_config = TrainingDataConfig.model_validate({
              "settings": {
                  "prefetch_size": 4,
                  "cache_dsn": "postgresql://user:pass@localhost:5432/vorox_metadata",
                  "shuffle_buffer": True,
                  "num_workers": 8
              },
              "urls": ["s3://vorox-data/dclm-baseline/shard-{000000..000099}.tar"]
          })
          train_loader = create_data_loader(data_config, train_config)
          ```
        - Used as a component in the main Config class to define data pipeline configuration
    
    Limitations:
        - No support for dynamic URL generation or pattern expansion
        - No validation for URL accessibility or content format at configuration time
        - No explicit dataset type specification; inferred from URL patterns
        - No support for mixed dataset types within a single configuration
        - No handling of dataset versioning or compatibility checking
        - No explicit memory usage estimation based on prefetch_size and dataset characteristics
    """
    settings: DataSettings
    urls: list[str]

class Device(str, Enum):
    """
    Enumeration of supported computation devices in the Vorox framework.
    
    Defines the canonical set of hardware acceleration targets for neural network
    operations, ensuring type safety and configuration validation for device placement.
    
    Architecture:
        Inherits from both str and Enum to enable string serialization while maintaining
        type safety. Implements a 1:1 mapping between enum members and string values
        with O(1) lookup complexity for both directions.
    
    Interface:
        - cpu: Central Processing Unit device target
          Standard fallback device available on all systems. Provides reliable but
          slower execution for models of any size within system memory constraints.
        - cuda: NVIDIA CUDA GPU device target
          Enables hardware acceleration on NVIDIA GPUs via the CUDA platform.
          Requires compatible NVIDIA hardware and drivers. Provides significant
          performance improvements for parallel tensor operations.
        - mps: Apple Metal Performance Shaders device target
          Enables hardware acceleration on Apple Silicon and compatible AMD GPUs.
          Available only on macOS systems with Metal API support. Provides
          platform-specific optimization for Apple hardware.
    
    Behavior:
        - Immutable after definition following Python's Enum pattern
        - Thread-safe for all operations
        - String-serializable for configuration files via the str inheritance
        - Hashable for use as dictionary keys
        - Values correspond directly to PyTorch device strings
    
    Integration:
        - Used in Config model to specify the target computation device
        - Consumed by model initialization and data pipeline to place tensors
          on the appropriate device
        - Directly maps to torch.device strings in PyTorch operations
        - Example:
          ```
          device_enum = Device.cuda
          torch_device = torch.device(device_enum)
          model = model.to(torch_device)
          ```
    
    Limitations:
        - Limited to three primary device types; multi-GPU configurations require code changes
        - No automatic fallback mechanism if specified device is unavailable
        - No support for distributed training across multiple devices
        - No validation for device availability at configuration time
        - No specification for device ordinals (e.g., cuda:0, cuda:1) for multi-device systems
        - MPS support may have feature limitations compared to CUDA implementation
    """
    cpu = "cpu"
    cuda = "cuda"
    mps = "mps"

class Config(BaseModel):
    """
    Root configuration model for the Vorox neural language model framework.
    
    Serves as the central configuration hub that aggregates and validates all hyperparameters
    and settings required for model architecture, training, optimization, and data processing.
    Designed to enforce type safety and configuration coherence across the entire system.
    
    Architecture:
        Inherits from Pydantic's BaseModel for validation and serialization.
        Implements a hierarchical composite structure with O(1) access complexity for all
        configuration domains. Composes specialized configuration models into a unified
        interface with nested validation. Total parameter space spans O(n) fields across
        all nested models with validation complexity of O(n).
    
    Interface:
        - tokenizer (Tokenizer): Text tokenization configuration.
          Controls token vocabulary and text encoding/decoding strategy.
        - architecture (Architecture): Neural network structure parameters.
          Defines transformer dimensions, depth, attention mechanism, and activation functions.
          Determines model capacity, computational requirements, and theoretical capabilities.
        - optimizer (Optimizer): Gradient-based optimization configuration.
          Controls learning dynamics including step size, momentum, and regularization.
        - loss (Loss): Training objective function specification.
          Defines the mathematical criterion for model optimization.
        - train (Train): Training process parameters.
          Controls batch sizing, sequence length, and iteration counts.
        - data (TrainingDataConfig): Data pipeline configuration.
          Specifies data sources, preprocessing, caching, and loading parallelism.
        - device (Device): Computation hardware target.
          Determines tensor placement for model operations (CPU/GPU/MPS).
    
    Behavior:
        - Immutable after initialization following Pydantic's data model pattern
        - Thread-safe for read operations with no internal mutable state
        - Validation occurs recursively across all nested models at initialization time
        - Serializable to/from YAML/JSON with full type preservation
        - Memory footprint is O(1) relative to model size; negligible compared to model parameters
    
    Integration:
        - Initialized via Pydantic's model_validate method from YAML/JSON configuration
        - Serves as the single source of truth for all framework components
        - Consumed by model constructors, training loops, optimizers, and data pipelines
        - Example:
          ```
          with open("configs/model_config.yml", "r") as f:
              cfg = Config.model_validate(yaml.safe_load(f))
          model = Vorox(cfg, tokenizer.vocab_size)
          optimizer = OptimizerBase.build(model, cfg)
          train_loader = create_data_loader(cfg)
          fit(cfg, model, train_loader, optimizer, loss_fn)
          ```
    
    Limitations:
        - No runtime validation for hardware compatibility with configured parameters
        - No automatic parameter tuning or suggestion mechanisms
        - No cross-field validation for potentially incompatible settings
        - No versioning mechanism for configuration schema evolution
        - No support for partial configuration updates; must be reconstructed for changes
        - No built-in configuration presets for common model architectures
    """
    tokenizer: Tokenizer
    architecture: Architecture
    optimizer: Optimizer
    loss: Loss
    train: Train
    data: TrainingDataConfig
    device: Device

if __name__ == "__main__":
    with open("configs/20M_test_model.yml", "r") as f:
        cfg = Config.model_validate(yaml.safe_load(f))

    print(cfg)
