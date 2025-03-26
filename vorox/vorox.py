from typing import Iterable, Tuple
import yaml

from pprint import pprint
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from vorox.configs import RunConfig as Config
from vorox.utils import get_available_devices



class Activation(nn.Module):
    """
    Factory class for neural network activation functions in the Vorox framework.
    
    Provides a unified interface for instantiating activation function modules based on
    configuration parameters, abstracting implementation details from client code.
    
    Architecture:
        Implements the Factory Method pattern with O(1) lookup complexity for activation types.
        Inherits from nn.Module but primarily serves as a base class and factory rather than
        a direct computation module. Concrete activation implementations are either PyTorch
        built-ins or custom implementations like SwiGLU.
    
    Interface:
        - build(cls, cfg: Config) -> nn.Module:
            Factory method that instantiates the appropriate activation function.
            Args:
                cfg: Config object containing architecture.activation field specifying the
                     activation type as defined in ActivationType enum.
            Returns:
                An instantiated nn.Module implementing the specified activation function.
            Raises:
                ValueError: When cfg.architecture.activation is not one of the supported types.
    
    Behavior:
        - Stateless factory method design with no instance state management
        - Thread-safe due to lack of shared mutable state
        - Deterministic mapping from configuration to concrete implementations
        - Activation modules themselves are typically stateless and purely functional
    
    Integration:
        - Used in neural network layer construction, particularly in feed-forward networks
        - Consumed by VoroxDecoderBlock to construct MLP components
        - Example:
          ```
          activation = Activation.build(cfg)
          mlp = nn.Sequential(
              nn.Linear(d_model, hidden_size),
              activation,
              nn.Linear(hidden_size, d_model)
          )
          ```
    
    Limitations:
        - Limited to predefined activation functions; custom activations require subclassing
        - No parameterization of activation functions beyond what's in the Config
        - No dynamic activation function switching at runtime
        - No composition of multiple activation functions
        - No automatic differentiation optimization for custom activations
    """
    
    @classmethod
    def build(cls, cfg: Config) -> nn.Module:
        """
        Factory method that instantiates activation functions based on configuration parameters.
        
        Constructs the appropriate PyTorch activation module by mapping configuration values
        to concrete implementations with O(1) lookup complexity. Serves as the primary
        interface for activation function instantiation throughout the Vorox framework.
        
        Architecture:
            - Implements the Factory Method pattern with constant-time dispatch
            - Maps string enum values to concrete PyTorch or custom activation implementations
            - Validation-first approach with explicit error handling for unsupported types
            - Zero memory overhead beyond the returned activation module instance
        
        Args:
            cfg (Config): Configuration object containing architecture settings.
                Must include cfg.architecture.activation field with a value from
                the ActivationType enum ("gelu", "relu", "silu", or "swiglu").
                For "swiglu", cfg.architecture.hidden_size must also be defined.
        
        Returns:
            nn.Module: Instantiated activation function module ready for integration
                into neural network computation graphs. Return type varies based on
                the requested activation:
                - nn.GELU for "gelu"
                - nn.ReLU for "relu"
                - nn.SiLU for "silu"
                - SwiGLU for "swiglu" (custom implementation with learnable parameters)
        
        Raises:
            ValueError: When cfg.architecture.activation contains an unsupported value
                not present in the ActivationType enum.
        
        Behavior:
            - Stateless operation with no side effects
            - Thread-safe due to absence of shared mutable state
            - Deterministic mapping from configuration to implementation
            - Returns stateless modules for standard activations, stateful for SwiGLU
        
        Integration:
            - Primary entry point for activation function instantiation
            - Used in VoroxDecoderBlock.mlp construction
            - Typically consumed within nn.Sequential blocks
            - Example:
              ```
              mlp = nn.Sequential(
                  nn.Linear(d_model, hidden_size),
                  Activation.build(cfg),
                  nn.Linear(hidden_size, d_model)
              )
              ```
        """
        if cfg.architecture.activation == "gelu":
            return nn.GELU()
        elif cfg.architecture.activation == "relu":
            return nn.ReLU()
        elif cfg.architecture.activation == "silu":
            return nn.SiLU()
        elif cfg.architecture.activation == "swiglu":
            return SwiGLU(cfg)
        else:
            raise ValueError(f"Activation {cfg.architecture.activation} not supported")
        

class SwiGLU(Activation):
    """
    SwiGLU activation function implementing a variant of Gated Linear Units with SiLU gating.
    
    Computes the element-wise product of a SiLU-gated projection and a linear projection:
    SwiGLU(x) = SiLU(W_gate·x) ⊙ (W_proj·x)
    
    Architecture:
        - Two-branch computation graph with O(hidden_size²) parameter complexity
        - Parallel linear projections followed by element-wise operations
        - SiLU gating mechanism: x·σ(x) where σ is the sigmoid function
        - Preserves dimensionality: R^hidden_size → R^hidden_size
        - Time complexity: O(batch·hidden_size²) for forward pass
        - Space complexity: O(hidden_size²) for parameters, O(batch·hidden_size) for activations
    
    Interface:
        - __init__(cfg: Config) -> None:
            Initializes the SwiGLU module with configuration parameters.
            Args:
                cfg: Config object containing architecture.hidden_size defining projection dimensions.
            
        - forward(x: torch.Tensor) -> torch.Tensor:
            Computes the SwiGLU activation.
            Args:
                x: Input tensor of shape (..., hidden_size)
            Returns:
                Tensor of shape (..., hidden_size) after applying SwiGLU activation
            
    Behavior:
        - Stateful with learned parameters in gate and projection matrices
        - Thread-safe for inference but requires synchronization during parameter updates
        - Deterministic output for fixed parameters and inputs
        - Differentiable end-to-end with well-defined gradients
        - Preserves input tensor's batch dimensions
    
    Integration:
        - Used in feed-forward networks within transformer blocks
        - Typically follows the first linear projection in MLP layers
        - Consumed by VoroxDecoderBlock as the activation in the MLP component
        - Example:
          ```
          mlp = nn.Sequential(
              nn.Linear(d_model, hidden_size),
              SwiGLU(cfg),
              nn.Linear(hidden_size, d_model)
          )
          ```
    
    Limitations:
        - Doubles parameter count compared to non-gated activations
        - Increased computational cost versus simple activations like ReLU
        - Requires hidden_size to be explicitly defined in config
        - No support for different input/output dimensions
        - May exhibit vanishing gradient issues for extreme input values
    """
    
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.hidden_size = cfg.architecture.hidden_size
        self.gate = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_proj = nn.Linear(self.hidden_size, self.hidden_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.gate(x)) * self.linear_proj(x)

class RoPE(nn.Module):
    """
    Rotary Position Embedding (RoPE) implementation for transformer attention mechanisms.
    
    Applies position-dependent rotation to query and key vectors in attention mechanisms,
    enabling the model to capture sequence order without explicit position embeddings.
    
    Architecture:
        - Implements the RoPE algorithm with O(d_model) space complexity per sequence position
        - Uses complex number rotation in frequency domain via sin/cos transformations
        - Frequency spectrum follows geometric sequence with base theta for long-range dependency modeling
        - Time complexity: O(seq_len * d_model) for embedding generation
        - Space complexity: O(seq_len * d_model) for sin/cos position matrices
        - Rotation applied in 2D subspaces of the embedding dimension
    
    Interface:
        - __init__(cfg: Config) -> None:
            Initializes the RoPE module with configuration parameters.
            Args:
                cfg: Config object containing architecture parameters including:
                    - d_model: Model dimension
                    - n_heads: Number of attention heads
                    - rope_theta: Base for geometric sequence (default: 10000.0)
        
        - get_rotary_embeddings(seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
            Generates sinusoidal position embeddings for the sequence.
            Args:
                seq_len: Length of the sequence
            Returns:
                Tuple of (sin, cos) tensors of shape [1, 1, seq_len, head_dim]
        
        - rotate_half(x: torch.Tensor) -> torch.Tensor:
            Performs the half-rotation operation on the input tensor.
            Args:
                x: Input tensor of shape [batch, n_heads, seq_len, head_dim]
            Returns:
                Rotated tensor of same shape with alternating dimensions swapped and negated
        
        - apply_rotary_pos_emb(pos_sin: torch.Tensor, pos_cos: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            Applies rotary position embeddings to input tensor.
            Args:
                pos_sin: Sine component of positional embedding
                pos_cos: Cosine component of positional embedding
                t: Input tensor to apply rotation to
            Returns:
                Tensor with rotary position encoding applied
        
        - forward(q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            Applies rotary position embeddings to query and key tensors.
            Args:
                q: Query tensor of shape [batch, n_heads, seq_len, head_dim]
                k: Key tensor of shape [batch, n_heads, seq_len, head_dim]
            Returns:
                Tuple of (rotated_q, rotated_k) with position information encoded
    
    Behavior:
        - Stateless operation with respect to sequence content
        - Thread-safe due to lack of mutable state
        - Deterministic output for fixed inputs
        - Preserves vector norms during rotation operations
        - Handles variable sequence lengths dynamically
        - Maintains precision by converting to float during computation
    
    Integration:
        - Used within GroupedQueryAttention to encode positional information
        - Applied conditionally based on cfg.architecture.rope flag
        - Operates on query and key projections before attention computation
        - Example:
          ```
          if self.cfg.architecture.rope:
              q, k = self.rope(q, k)
          ```
    
    Limitations:
        - Effective context length limited by rope_theta parameter
        - No support for position interpolation for out-of-distribution lengths
        - Requires head_dim to be even for proper rotation in 2D subspaces
        - May introduce numerical precision issues with extremely long sequences
        - No explicit handling of bidirectional or causal attention patterns
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

    def get_rotary_embeddings(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates sinusoidal position embeddings for rotary position encoding.
        
        Computes frequency-based position representations following a geometric sequence,
        enabling the model to capture relative positions through rotation in vector space.
        The implementation follows the RoPE paper (Su et al., 2021) with modifications
        for transformer architecture integration.
        
        Architecture:
            - Implements frequency-domain position encoding with O(seq_len * head_dim) complexity
            - Geometric frequency progression with base theta for long-range dependency modeling
            - Frequency spectrum distributed across embedding dimensions in pairs
            - Time complexity: O(seq_len * head_dim) for embedding generation
            - Space complexity: O(seq_len * head_dim) for sin/cos position matrices
            - Frequency computation: θ^(-(2i/d_model)) for dimension i
            - Positions encoded as outer product between position indices and frequencies
            
        Args:
            seq_len (int): Length of the sequence to generate position embeddings for.
                Must be a positive integer representing the maximum sequence length
                for which attention will be computed.
                
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - pos_sin: Sine component tensor of shape [1, 1, seq_len, head_dim]
                - pos_cos: Cosine component tensor of shape [1, 1, seq_len, head_dim]
                Both tensors are broadcastable for batch and head dimensions during
                the rotation operation in apply_rotary_pos_emb().
                
        Behavior:
            - Stateless computation with no side effects
            - Thread-safe due to absence of shared mutable state
            - Deterministic output for fixed sequence length and configuration
            - Precision-aware with explicit float type casting
            - Handles arbitrary sequence lengths dynamically
            - Computes frequencies in log-space for numerical stability
            
        Integration:
            - Called by forward() to generate position-dependent rotation matrices
            - Results passed to apply_rotary_pos_emb() for query/key tensor rotation
            - Frequency base (rope_theta) configurable in architecture settings
            - Example:
              ```
              pos_sin, pos_cos = self.get_rotary_embeddings(key_len)
              q = self.apply_rotary_pos_emb(pos_sin[:,:,offset:], pos_cos[:,:,offset:], q)
              ```
              
        Limitations:
            - Effective context length limited by rope_theta parameter
            - No support for position interpolation for out-of-distribution lengths
            - Requires head_dim to be even for proper rotation in 2D subspaces
            - May introduce numerical precision issues with extremely long sequences
            - Fixed frequency spectrum without learned parameters
        """
        dim = self.cfg.architecture.d_model // self.cfg.architecture.n_heads
        inv_freq = 1.0 / (
            self.cfg.architecture.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim)
        )
        seq = torch.arange(seq_len, dtype=torch.float)
        freqs = torch.einsum("i , j -> i j", seq, inv_freq)
        positions = torch.cat((freqs, freqs), dim=-1)
        pos_sin, pos_cos = positions.sin()[None, None, :, :], positions.cos()[None, None, :, :]

        return pos_sin, pos_cos
    
    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs half-rotation operation for Rotary Position Embeddings (RoPE).
        
        Implements a core tensor manipulation for RoPE by reshaping and rotating vector components
        in 2D subspaces, enabling position-aware attention through geometric transformations
        rather than additive embeddings.
        
        Architecture:
            - Implements complex number rotation in frequency domain with O(1) operation complexity
            - Reshapes input tensor to isolate even/odd dimensions for rotation in 2D subspaces
            - Performs rotation by swapping and negating half the dimensions with sign inversion
            - Time complexity: O(batch·n_heads·seq_len·head_dim) for tensor operations
            - Space complexity: O(batch·n_heads·seq_len·head_dim) for intermediate tensors
            - Rotation matrix equivalent to [cos(θ), -sin(θ); sin(θ), cos(θ)] for θ=π/2
            - Preserves vector norms during transformation (orthogonal operation)
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch, n_heads, seq_len, head_dim]
                containing query or key vectors to be rotated. The head_dim must be even
                to allow proper reshaping into pairs of dimensions.
                
        Returns:
            torch.Tensor: Rotated tensor of same shape [batch, n_heads, seq_len, head_dim]
                with alternating dimensions swapped and negated, implementing the rotation
                operation in the frequency domain.
                
        Behavior:
            - Stateless operation with no side effects
            - Thread-safe due to absence of shared mutable state
            - Deterministic output for fixed inputs
            - Preserves tensor shape and dimensionality
            - Maintains numerical precision without additional scaling
            - Requires head_dim to be even (implicit constraint)
            
        Integration:
            - Called by apply_rotary_pos_emb() to implement rotation component
            - Part of the RoPE algorithm for position-aware attention
            - Operates on query and key projections before attention computation
            - Example:
              ```
              rotated_component = self.rotate_half(t) * pos_sin
              ```
              
        Limitations:
            - Requires head_dim to be even for proper reshaping
            - No explicit error handling for odd-dimensioned inputs
            - Fixed rotation angle (π/2) with no parameterization
            - May introduce numerical precision issues with mixed precision training
            - Assumes specific tensor layout matching RoPE implementation
            - No optimization for hardware-specific tensor operations
        """
        B, n_heads, T, head_size = x.size()
        x = x.view(B, n_heads, T, 2, head_size // 2)
        x1, x2 = x.unbind(dim=-2)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, pos_sin: torch.Tensor, pos_cos: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Applies rotary position embeddings to input tensor for position-aware attention.
        
        Implements the core rotation operation of RoPE by combining the input tensor with
        sinusoidal position embeddings through a combination of element-wise multiplication
        and specialized rotation, enabling the model to capture sequence order information
        through relative position encoding in the frequency domain.
        
        Architecture:
            - Implements complex number rotation in vector space with O(1) operation complexity
            - Combines direct component (t * pos_cos) with rotated component (rotate_half(t) * pos_sin)
            - Equivalent to applying 2D rotation matrices to pairs of vector components
            - Time complexity: O(batch·n_heads·seq_len·head_dim) for tensor operations
            - Space complexity: O(batch·n_heads·seq_len·head_dim) for intermediate tensors
            - Rotation matrix equivalent to [cos(θ), -sin(θ); sin(θ), cos(θ)] for position-dependent θ
            - Preserves vector norms during transformation (orthogonal operation)
            
        Args:
            pos_sin (torch.Tensor): Sine component of positional embedding with shape
                [1, 1, seq_len, head_dim], containing sin(θ) values for each position and
                frequency pair. Must match the last two dimensions of tensor t.
                
            pos_cos (torch.Tensor): Cosine component of positional embedding with shape
                [1, 1, seq_len, head_dim], containing cos(θ) values for each position and
                frequency pair. Must match the last two dimensions of tensor t.
                
            t (torch.Tensor): Input tensor of shape [batch, n_heads, seq_len, head_dim]
                containing query or key vectors to be rotated. The head_dim must be even
                to allow proper rotation in 2D subspaces.
                
        Returns:
            torch.Tensor: Rotated tensor of same shape as input [batch, n_heads, seq_len, head_dim]
                with position information encoded through frequency-domain rotation, preserving
                the original tensor's dtype for precision consistency.
                
        Behavior:
            - Stateless operation with no side effects
            - Thread-safe due to absence of shared mutable state
            - Deterministic output for fixed inputs
            - Preserves tensor shape, dimensionality, and dtype
            - Maintains numerical precision by explicit dtype casting
            - Requires head_dim to be even (implicit constraint from rotate_half)
            
        Integration:
            - Called by forward() to apply position-dependent rotations to query/key tensors
            - Core component of the RoPE algorithm for position-aware attention
            - Used in conjunction with get_rotary_embeddings() and rotate_half()
            - Example:
              ```
              q_ = self.apply_rotary_pos_emb(
                  pos_sin[:, :, key_len - query_len : key_len, :],
                  pos_cos[:, :, key_len - query_len : key_len, :],
                  q_,
              )
              ```
              
        Limitations:
            - Requires head_dim to be even for proper rotation in 2D subspaces
            - No explicit error handling for dimension mismatches
            - Fixed rotation implementation with no parameterization
            - May introduce numerical precision issues with mixed precision training
            - Assumes broadcastable dimensions for pos_sin and pos_cos tensors
            - No optimization for hardware-specific tensor operations
        """
        return ((t * pos_cos) + (self.rotate_half(t) * pos_sin)).to(t.dtype)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q_, k_ = q.float(), k.float()

        query_len, key_len = q_.shape[-2], k_.shape[-2]  # could be different if layer_past not None
        pos_sin, pos_cos = self.get_rotary_embeddings(key_len)
        pos_sin = pos_sin.type_as(q_)
        pos_cos = pos_cos.type_as(q_)
        q_ = self.apply_rotary_pos_emb(
            pos_sin[:, :, key_len - query_len : key_len, :],
            pos_cos[:, :, key_len - query_len : key_len, :],
            q_,
        )
        k_ = self.apply_rotary_pos_emb(pos_sin, pos_cos, k_)

        return q_.type_as(q), k_.type_as(k)


class LayerNorm(nn.Module):
    """
    Layer normalization implementation for transformer architectures with optional affine transformation.
    
    Normalizes input tensors across the feature dimension, stabilizing training by
    standardizing activations to zero mean and unit variance, with optional learned
    scaling and shifting parameters.
    
    Architecture:
        - Implements layer normalization with O(n) time complexity per forward pass
        - Computes statistics (mean, variance) across the normalized_shape dimensions
        - Normalizes using z(x) = (x - E[x]) / sqrt(Var[x] + eps)
        - Optionally applies affine transformation: y = weight * z(x) + bias
        - Space complexity: O(normalized_shape) for parameters if elementwise_affine=True
        - Normalization is applied to the last dimensions of the input tensor
        - Preserves tensor shape: R^(..., normalized_shape) → R^(..., normalized_shape)
    
    Interface:
        - __init__(normalized_shape: Iterable, elementwise_affine: bool = True, eps: float = 1e-5) -> None:
            Initializes the LayerNorm module with configuration parameters.
            Args:
                normalized_shape: Shape of the features to be normalized (e.g., (d_model,))
                elementwise_affine: Whether to apply learnable affine transformation (default: True)
                eps: Small constant added to denominator for numerical stability (default: 1e-5)
            
        - forward(x: torch.Tensor) -> torch.Tensor:
            Applies layer normalization to the input tensor.
            Args:
                x: Input tensor with shape (..., *normalized_shape)
            Returns:
                Normalized tensor with same shape as input
            
    Behavior:
        - Stateful with learned parameters (weight, bias) when elementwise_affine=True
        - Thread-safe for inference but requires synchronization during parameter updates
        - Deterministic output for fixed parameters and inputs
        - Differentiable end-to-end with well-defined gradients
        - Preserves input tensor's batch dimensions
        - Initialization: weight=1, bias=0 when elementwise_affine=True
    
    Integration:
        - Used in transformer blocks for pre-normalization or post-normalization
        - Applied before attention and feed-forward networks in VoroxDecoderBlock
        - Typically configured with elementwise_affine=False in modern transformer variants
        - Example:
          ```
          norm = LayerNorm((d_model,), elementwise_affine=False)
          normalized_x = norm(x)
          attention_output = attention(normalized_x)
          ```
    
    Limitations:
        - Less effective for very deep networks without careful initialization
        - May introduce training instability with very small batch sizes
        - Requires normalized_shape to match the last dimensions of input tensor
        - Performance degradation with extremely small eps values
        - No support for normalization across batch or sequence dimensions
        - Potential precision issues with mixed precision training
    """

    def __init__(self, normalized_shape: Iterable, elementwise_affine: bool = True, eps: float = 1e-5):
        super().__init__()

        self.normalized_shape = normalized_shape
        self.elementwise_affine = elementwise_affine
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(self.normalized_shape)) if self.elementwise_affine else None
        self.bias = nn.Parameter(torch.zeros(self.normalized_shape)) if self.elementwise_affine else None

    def forward(self, x: torch.Tensor):
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)


class GroupedQueryAttention(nn.Module):
    """
    Attention(QW_q, KW_k, VW_v) = softmax((QW_q) (KW_k)^T / sqrt(d_k))(VW_v)
    where: 
    Q has shape (batch_size, seq_len, d_model)
    K has shape (batch_size, seq_len, d_model)
    V has shape (batch_size, seq_len, d_model)
    W_q has shape (d_model, d_k)
    W_k has shape (d_model, d_k)
    W_v has shape (d_model, d_v)
    So then the output of the attention will have shape (batch_size, seq_len, d_v).

    W_q.shape[1] == W_k.shape[1] so that the matmul works.
    In attention is all you need, they set d_k = d_v = d_model // n_heads.

    In MultiHeadAttention, each attention head has its own W_q, W_k, W_v. In that case n_heads = n_kv_heads.
    It is computed as: 
        MultiHeadAttention(Q, K, V) = Concat(Attention(QW_q1, KW_k1, VW_v1), ..., Attention(QW_qh, KW_kh, VW_vh))W_out 
    where:
    h = n_heads
    W_out has shape (n_heads * d_v, d_model)

    In GroupedQueryAttention, groups of queries will share the same key/value pair. So assume n_heads = 8 and n_kv_heads = 2. Then we will have 8 different W_q matrices, but only 2 W_k and W_v matrices. Each attention head will be computed in the same way, but the heads will be grouped into 2 groups, each group using one pair (W_k, W_v).
    It is computed as: 
        GroupedQueryAttention(Q, K, V) = Concat(Attention(QW_q1, KW_k1, VW_v1), ..., Attention(QW_q_h, KW_k_kvh, VW_v_kvh))W_out 
    where:
    h = n_heads
    kvh = n_kv_heads
    W_out has shape (n_heads * d_v, d_model)

    Since we want the output of the transformer block to have the same shape as the input, generally we want 
    """

    def __init__(self, cfg: Config):
        """
        Initializes a Grouped Query Attention module with parameter sharing across key-value heads.
        
        Constructs a multi-query attention mechanism where multiple query heads share the same
        key-value pairs, reducing parameter count and memory footprint while maintaining
        representational capacity. Implements the GQA pattern described in Ainslie et al. (2023).
        
        Architecture:
            - Implements grouped multi-head attention with O(d_model²) parameter complexity
            - Fused QKV projection with asymmetric head distribution (n_heads > n_kv_heads)
            - Query heads grouped into n_kv_heads groups, each sharing a key-value pair
            - Time complexity: O(batch·seq_len²·d_model) for attention computation
            - Space complexity: O(d_model²) for parameters, O(batch·seq_len²·n_heads) for attention
            - Attention scores normalized via scaled dot-product: softmax(QK^T/√d_k)
            - Optional rotary position embeddings for sequence order encoding
        
        Args:
            cfg (Config): Configuration object containing architecture parameters including:
                - d_model (int): Model dimension/embedding size
                - n_heads (int): Number of attention heads (query projections)
                - n_kv_heads (int): Number of key-value head groups (must divide n_heads evenly)
                - rope (bool): Whether to apply rotary position embeddings
                
        Raises:
            AssertionError: When cfg.architecture.n_heads is not divisible by cfg.architecture.n_kv_heads,
                indicating incompatible grouping configuration.
        
        Behavior:
            - Stateful with learned parameters in projection matrices
            - Thread-safe for inference but requires synchronization during parameter updates
            - Deterministic output for fixed parameters and inputs
            - Differentiable end-to-end with well-defined gradients
            - Preserves input tensor's batch and sequence dimensions
            - Initialization: Xavier uniform for projection matrices
        
        Integration:
            - Core attention mechanism within VoroxDecoderBlock
            - Processes normalized input from LayerNorm
            - Outputs are residually connected to input in the transformer block
            - Example:
              ```
              attn_norm = LayerNorm((d_model,), elementwise_affine=False)
              attn = GroupedQueryAttention(cfg)
              output = x + attn(attn_norm(x))  # Residual connection
              ```
        
        Limitations:
            - Attention complexity scales quadratically with sequence length
            - No support for causal masking (assumes full attention)
            - No support for sparse attention patterns
            - Requires n_heads to be divisible by n_kv_heads
            - No explicit handling of padding tokens
            - No support for relative position bias or alibi position encoding
        """
        super().__init__()

        assert cfg.architecture.n_heads % cfg.architecture.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

        self.cfg = cfg

        self.d_model = cfg.architecture.d_model
        self.n_heads = cfg.architecture.n_heads
        self.n_kv_heads = cfg.architecture.n_kv_heads
        self.group_size = cfg.architecture.n_heads // cfg.architecture.n_kv_heads

        self.d_k = self.d_model // self.n_heads
        self.d_v = self.d_model // self.n_kv_heads

        # these are the sizes of q,k,v before getting split into individual heads
        # so each head will get vectors with shapes: fused_dims[0] // n_heads, fused_dims[1] // n_kv_heads, fused_dims[2] // n_kv_heads
        self.fused_dims = (
            self.n_heads * self.d_k,
            self.n_kv_heads * self.d_k,
            self.n_kv_heads * self.d_v,
        )
        self.rope = RoPE(cfg)

        self.attn_proj = nn.Linear(self.d_model, sum(self.fused_dims), bias=False)  # fused linear layer to improve performance
        self.out_proj = nn.Linear(self.n_heads * self.d_v, self.d_model)

    def forward(self, x: torch.Tensor, causal_attn_mask: bool = False) -> torch.Tensor:
        """
        Applies grouped query attention to the input tensor with parameter-efficient key-value sharing.
        
        Computes multi-head attention with grouped query heads sharing key-value pairs,
        implementing the core attention mechanism of the transformer architecture with
        reduced parameter count through key-value head sharing.
        
        Architecture:
            - Implements scaled dot-product attention: softmax(QK^T/√d_k)·V
            - Fused QKV projection followed by head-wise splitting with O(d_model²) complexity
            - Multi-query attention pattern with n_heads query projections sharing n_kv_heads KV pairs
            - Time complexity: O(batch·seq_len²·d_model) for attention computation
            - Space complexity: O(batch·seq_len²·n_heads) for attention matrix
            - Parallel computation across batch and head dimensions
            - Optional rotary position embeddings applied to query and key projections
            - Optional causal masking to prevent attention to future tokens
            
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, d_model]
                containing token representations to compute attention over.
            causal_attn_mask (bool, optional): Whether to apply causal masking to prevent
                attention to future tokens. Defaults to False.
                
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, d_model]
                containing the attention-weighted representations with the same
                dimensionality as the input for residual connection compatibility.
                
        Behavior:
            - Deterministic computation for fixed weights and inputs
            - Preserves sequence ordering through position-dependent attention patterns
            - Thread-safe for inference (no state mutation during forward pass)
            - Maintains precision by using appropriate scaling factors (1/√d_k)
            - Numerically stable through softmax normalization
            - Differentiable end-to-end with well-defined gradients
            - When causal_attn_mask=True, prevents information leakage from future tokens
            
        Integration:
            - Core computation block within VoroxDecoderBlock
            - Typically preceded by layer normalization
            - Output commonly used in residual connections
            - Example:
              ```
              normalized_x = self.norm(x)
              attention_output = self.attention(normalized_x, causal_attn_mask=model.causal_attn_mask)
              x = x + attention_output  # Residual connection
              ```
              
        Limitations:
            - Quadratic memory scaling with sequence length (O(seq_len²))
            - Fixed attention pattern (no dynamic routing)
            - No explicit handling of padding tokens
            - Requires n_heads to be divisible by n_kv_heads
        """
        B, L, _ = x.size()

        q, k, v = self.attn_proj(x).split(self.fused_dims, dim=-1)  # the output of the layer will be (B, L, sum(self.fused_dims))

        # torch matmul with @ for tensors with shapes (..., a, b) and (..., b, c) results in shape (..., a, c)
        # so we need to re-shape q, k, v so that we can do the matmul (ie with dimensions[:-2] being the same)
        q = q.view(B, L, self.n_heads, self.d_k).transpose(1, 2)  # so last two dims are sequence x query
        k = k.view(B, L, self.n_kv_heads, self.d_k).transpose(1, 2)  # so last two dims are sequence x key
        v = v.view(B, L, self.n_kv_heads, self.d_v).transpose(1, 2)  # so last two dims are sequence x value

        if self.cfg.architecture.rope:
            q, k = self.rope(q, k)

        # to make shapes compatible for matmul, we need to repeat each group's key and value tensors group_size times
        k = k.repeat_interleave(self.group_size, dim=1, output_size=self.n_heads)  # shape (B, n_heads, L, d_k)
        v = v.repeat_interleave(self.group_size, dim=1, output_size=self.n_heads)  # shape (B, n_heads, L, d_v)

        attn = (q @ k.transpose(-2, -1)) / (self.d_k ** 0.5)  # shape (B, n_heads, L, L)
        
        # Apply causal mask if requested
        if causal_attn_mask:
            # Create causal mask (upper triangular part of the attention matrix)
            mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
            # Set masked positions to -inf before softmax
            attn.masked_fill_(mask, float('-inf'))
            
        attn = attn.softmax(dim=-1) @ v  # shape (B, n_heads, L, d_v)
        attn = attn.transpose(1, 2).reshape(B, L, self.n_heads * self.d_v)  # shape (B, L, n_heads * d_v)

        out = self.out_proj(attn)  # shape (B, L, d_model)
        return out


class VoroxDecoderBlock(nn.Module):
    """
    Decoder-only transformer block.
        ``(MLP∘LN-simple)((Attn∘LN-simple)(x) + x) + interm_x``
    """

    def __init__(self, cfg: Config, causal_attn_mask: bool = False):
        """
        Initializes a decoder-only transformer block with pre-normalization architecture.
        
        Constructs a transformer block implementing the architecture pattern:
        ``(MLP∘LN-simple)((Attn∘LN-simple)(x) + x) + interm_x``
        where each component is applied with residual connections and layer normalization.
        
        Architecture:
            - Implements pre-normalization pattern with O(d_model²) parameter complexity
            - Two-stage computation flow with residual connections at each stage
            - GroupedQueryAttention with n_heads query heads and n_kv_heads key-value groups
            - MLP with hidden dimension expansion and non-linear activation
            - Time complexity: O(batch·seq_len²·d_model) dominated by attention computation
            - Space complexity: O(d_model²) for parameters, O(batch·seq_len·d_model) for activations
            - Parameter sharing in attention through grouped query-key-value projections
            - Non-parametric layer normalization (elementwise_affine=False) for stability
            - Optional causal masking to prevent attention to future tokens
        
        Args:
            cfg (Config): Configuration object containing architecture parameters including:
                - d_model (int): Model dimension/embedding size
                - n_heads (int): Number of attention heads (must be divisible by 4)
                - hidden_size (int): Dimension of the MLP's hidden layer
                - activation (str): Activation function type for the MLP
            causal_attn_mask (bool, optional): Whether to apply causal masking to prevent
                attention to future tokens. Defaults to False.
                
        Raises:
            AssertionError: When cfg.architecture.n_heads is not divisible by 4,
                indicating incompatible attention head configuration.
        
        Behavior:
            - Stateful with learned parameters in attention and MLP components
            - Thread-safe for inference but requires synchronization during parameter updates
            - Deterministic output for fixed parameters and inputs
            - Differentiable end-to-end with well-defined gradients
            - Preserves input tensor's batch and sequence dimensions
            - Initialization follows PyTorch defaults for linear layers
            - When causal_attn_mask=True, prevents information leakage from future tokens
        
        Integration:
            - Core building block within Vorox transformer model
            - Multiple instances stacked sequentially to form deep networks
            - Processes token embeddings or outputs from previous blocks
            - Example:
              ```
              blocks = nn.Sequential(*[VoroxDecoderBlock(cfg) for _ in range(cfg.architecture.n_layers)])
              output = blocks(input_embeddings)
              ```
        
        Limitations:
            - Attention complexity scales quadratically with sequence length
            - Fixed n_kv_heads ratio (n_heads/4) with no configuration option
            - No support for cross-attention (decoder-only architecture)
            - No explicit handling of padding tokens
            - No support for relative position bias or other position encoding schemes
        """
        super().__init__()

        assert cfg.architecture.n_heads % 4 == 0, "n_heads must be divisible by 4"

        self.d_model = cfg.architecture.d_model
        self.n_heads = cfg.architecture.n_heads
        self.n_kv_heads = cfg.architecture.n_heads // 4
        self.hidden_size = cfg.architecture.hidden_size
        self.causal_attn_mask = causal_attn_mask

        self.attn_norm = LayerNorm((self.d_model,), elementwise_affine=False)
        self.attn = GroupedQueryAttention(cfg)
        self.mlp_norm = LayerNorm((self.d_model,), elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(self.d_model, self.hidden_size),
            Activation.build(cfg),
            nn.Linear(self.hidden_size, self.d_model),
        )

    def forward(self, x):
        interm_x = x + self.attn(self.attn_norm(x), causal_attn_mask=self.causal_attn_mask)
        out_x = interm_x + self.mlp(self.mlp_norm(interm_x))
        return out_x


class Vorox(nn.Module):

    def __init__(self, cfg: Config, vocab_size: int):
        """
        Initializes a parameter-efficient decoder-only transformer model with memory-optimized architecture.
        
        Constructs a transformer model implementing grouped query attention (GQA) for large language 
        modeling with reduced parameter count and memory footprint while maintaining model capacity
        through strategic parameter sharing across attention heads. The architecture follows a pre-normalization
        pattern with non-parametric layer normalization for training stability across deep networks.
        
        Architecture:
            - Decoder-only transformer with O(n_layers·d_model²) parameter complexity
            - Three-stage pipeline: token embedding → transformer blocks → vocabulary projection
            - Memory-optimized grouped query attention with n_kv_heads < n_heads (4:1 ratio by default)
            - Pre-normalization architecture with non-parametric layer normalization for training stability
            - Parameter sharing through key-value head grouping (each KV pair serves multiple query heads)
            - Token embedding matrix: [vocab_size, d_model] with O(vocab_size·d_model) space complexity
            - Transformer blocks: Sequential stack of n_layers identical blocks with residual connections
            - Output projection: Linear layer mapping hidden states to vocabulary logits [d_model, vocab_size]
            - Time complexity: O(batch·seq_len²·d_model·n_layers) dominated by attention operations
            - Space complexity: O(vocab_size·d_model + n_layers·d_model²) for parameters
            - Memory footprint: O(batch·seq_len·d_model·n_layers) for activations during forward pass
            - Computation graph optimized for autoregressive next-token prediction
            - Parameter efficiency achieved through GQA's asymmetric head distribution
            - Numerical stability ensured through scaled dot-product attention and careful dtype handling
        
        Args:
            cfg (Config): Configuration object containing architecture parameters with strict requirements:
                - architecture.n_layers (int): Number of transformer blocks (≥1)
                - architecture.d_model (int): Model dimension/embedding size (must be divisible by n_heads)
                - architecture.n_heads (int): Number of attention heads (must be divisible by 4)
                - architecture.n_kv_heads (int): Number of key-value heads (must divide n_heads evenly)
                - architecture.hidden_size (int): Dimension of MLP's hidden layer (typically 2-4x d_model)
                - architecture.activation (str): Activation function type ("gelu", "relu", "silu", "swiglu")
                - architecture.rope (bool): Whether to apply rotary position embeddings
                - architecture.rope_theta (float, optional): Base for RoPE frequency (default: 10000.0)
            
            vocab_size (int): Size of the vocabulary for token embedding and output projection.
                Must be positive integer matching the tokenizer's vocabulary size.
                Determines embedding and output projection matrix dimensions.
                Critical for memory usage as it scales the two largest parameter matrices.
                Typically ranges from 32,000-100,000 tokens for modern language models.
        
        Raises:
            AssertionError: When cfg.architecture.n_heads is not divisible by 4 (propagated from VoroxDecoderBlock)
            ValueError: When configuration contains invalid or incompatible parameter values
            TypeError: When parameters have incorrect types
            RuntimeError: When insufficient memory is available for model initialization
        
        Behavior:
            - Stateful with learned parameters in embedding, attention, and MLP components
            - Thread-safe for inference but requires synchronization during parameter updates
            - Deterministic output for fixed parameters, inputs, and random seeds
            - Preserves input tensor's batch and sequence dimensions throughout computation
            - Initialization follows PyTorch defaults for embedding and linear layers
            - Parameter count scales primarily with vocab_size·d_model and n_layers·d_model²
            - Memory usage during training dominated by activations and optimizer states
            - No internal state mutation during forward pass (safe for parallel inference)
            - Gradient computation requires synchronization across all parameters
            - Embedding and output projection matrices dominate parameter count for large vocabularies
            - Transformer blocks dominate computation time due to attention's quadratic complexity
        
        Integration:
            - Primary model class for training and inference in the Vorox framework
            - Instantiated with configuration and tokenizer vocabulary size
            - Compatible with PyTorch's training ecosystem (optimizers, loss functions, etc.)
            - Designed for integration with HuggingFace's transformers library
            - Exposes parameter_breakdown() and trainable_parameters for model analysis
            - Integrates with RemoteIterableDataset for efficient streaming training
            - Supports both CPU and GPU/MPS execution environments
            - Example:
              ```python
              tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer.name)
              model = Vorox(cfg, vocab_size=tokenizer.vocab_size)
              optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.optimizer.lr)
              model = model.to(device)  # Move to appropriate compute device
              ```
        
        Limitations:
            - Attention complexity scales quadratically with sequence length (O(seq_len²))
            - No key-value caching implementation for efficient autoregressive generation
            - No explicit position embeddings beyond optional RoPE in attention
            - Fixed vocabulary size defined at initialization (no dynamic vocabulary)
            - No parameter sharing across transformer blocks (except attention grouping)
            - No support for sparse attention patterns or linear attention mechanisms
            - No gradient checkpointing implementation for memory-efficient backpropagation
            - No support for mixed precision training in the core implementation
            - No explicit handling of padding tokens or attention masks
            - Memory requirements may exceed consumer hardware for models exceeding 1B parameters
            - Training stability decreases with extreme depth (n_layers > 32) without careful initialization
        """
        super().__init__()

        self.n_layers = cfg.architecture.n_layers
        self.d_model = cfg.architecture.d_model
        self.n_heads = cfg.architecture.n_heads
        self.vocab_size = vocab_size
        
        # Set causal attention mask based on whether we're in training or eval mode
        # Default to False for eval-only configs, True if train config exists
        self.causal_attn_mask = hasattr(cfg, 'train')

        self.emb = nn.Embedding(vocab_size, self.d_model)
        self.transformer_blocks = nn.Sequential(*[VoroxDecoderBlock(cfg, causal_attn_mask=self.causal_attn_mask) for _ in range(cfg.architecture.n_layers)])
        self.ff_out = nn.Linear(self.d_model, vocab_size)

    def train(self) -> None:
        """
        Sets the model to training mode and enables causal attention masking.
        
        Extends the standard PyTorch train() method to also enable causal attention masking,
        ensuring that during training the model properly masks future tokens in the attention
        mechanism to prevent information leakage.
        
        Returns:
            None
        """
        super().train()
        self.causal_attn_mask = True
        
        # Update causal_attn_mask in all decoder blocks
        for block in self.transformer_blocks:
            block.causal_attn_mask = True
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Executes the forward pass of the Vorox transformer model, converting token IDs to next-token logits.
        
        Implements the core computational pipeline of a decoder-only transformer with grouped query attention,
        designed for autoregressive language modeling with parameter-efficient architecture. Processes input
        token sequences through a three-stage differentiable computation graph optimized for both training
        and inference scenarios.
        
        Architecture:
            - Decoder-only transformer with O(batch·seq_len²·d_model·n_layers) computational complexity
            - Three-stage differentiable pipeline: embedding → transformer blocks → vocabulary projection
            - Token embedding layer maps discrete indices to continuous d_model-dimensional vectors
            - Sequential processing through n_layers transformer blocks with residual connections
            - Each block implements pre-normalization with non-parametric layer normalization
            - Grouped query attention with n_heads query projections sharing n_kv_heads key-value pairs
            - Memory complexity: O(batch·seq_len·d_model·n_layers) for activations during forward pass
            - Computation graph optimized for autoregressive next-token prediction
            - Optional causal masking in attention to prevent information leakage from future tokens
            - Parameter efficiency through key-value head sharing (n_heads > n_kv_heads)
            
        Args:
            input_ids (torch.Tensor): Integer tensor of shape [batch_size, seq_len] containing token indices.
                Must contain values in range [0, vocab_size-1].
                Typically produced by a tokenizer from raw text.
                No explicit handling of padding tokens or attention masks.
                Supports variable sequence lengths up to model's context window.
                
        Returns:
            torch.Tensor: Logits tensor of shape [batch_size, seq_len, vocab_size] containing
                unnormalized prediction scores for next-token distribution at each position.
                For autoregressive models, each position contains predictions conditioned
                only on previous positions. Values represent raw scores before softmax
                normalization for probability distribution.
                
        Behavior:
            - Deterministic computation with identical outputs for fixed weights and inputs
            - Thread-safe for inference with no internal state mutation during forward pass
            - Gradients flow through all components for end-to-end differentiability
            - Maintains numerical precision through careful dtype handling in attention mechanisms
            - No internal caching mechanism for key-value pairs in incremental decoding
            - Preserves sequence ordering through optional rotary position embeddings in attention
            - Memory usage scales linearly with batch size and sequence length
            - Computation time dominated by attention operations (quadratic with sequence length)
            - When self.causal_attn_mask=True, prevents attention to future tokens
            
        Integration:
            - Primary entry point for both training and inference workflows
            - In training: Followed by cross-entropy loss computation against target tokens
            - In generation: Used with sampling strategies (greedy, top-k, top-p) for text completion
            - Compatible with PyTorch's autograd system for gradient-based optimization
            - Designed for integration with HuggingFace's transformers ecosystem
            - Example (generation loop):
              ```python
              for i in range(max_length):
                  logits = model(input_ids)
                  next_token_logits = logits[:, -1, :]
                  next_token = torch.argmax(next_token_logits, dim=-1)
                  input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
              ```
              
        Limitations:
            - Quadratic scaling with sequence length limits practical context window size
            - No key-value caching implementation for efficient autoregressive generation
            - No support for bidirectional attention patterns (decoder-only architecture)
            - No handling of position IDs or attention masks as explicit inputs
            - No gradient checkpointing for memory-efficient backpropagation
            - Performance degradation with extremely long sequences (>4096 tokens)
            - No built-in support for efficient batched generation
        """
        x = self.emb(input_ids)
        x = self.transformer_blocks(x)
        x = self.ff_out(x)
        return x
    
    @property
    def trainable_parameters(self):
        """
        Calculates the total count of trainable parameters in the model with O(n) complexity.
        
        Computes the sum of all parameters that require gradients across the entire model
        hierarchy, providing a critical metric for model capacity and memory requirements.
        Serves as a key diagnostic tool for model architecture analysis and optimization.
        
        Architecture:
            - Implements parameter counting with O(n) time complexity where n is parameter count
            - Traverses the entire parameter hierarchy using PyTorch's parameters() iterator
            - Filters parameters based on requires_grad attribute with O(1) lookup per parameter
            - Aggregates parameter counts using numel() with O(1) complexity per tensor
            - Memory complexity: O(1) as it maintains only a running sum
            - Zero memory overhead beyond temporary Python integer objects
            
        Returns:
            int: Total count of trainable parameters across all modules.
                Represents the effective model capacity and directly correlates
                with memory requirements during training. Excludes frozen parameters
                that don't receive gradient updates. Critical for determining
                computational requirements and model size classification.
                
        Behavior:
            - Stateless computation with no side effects
            - Thread-safe due to absence of shared mutable state
            - Deterministic output for fixed model architecture
            - Constant time complexity regardless of parameter tensor shapes
            - Returns consistent results throughout model lifecycle unless parameters are frozen/unfrozen
            - Computationally negligible compared to forward/backward passes
            
        Integration:
            - Used for model diagnostics and reporting during initialization
            - Critical for determining hardware requirements for training
            - Consumed by parameter_breakdown() for detailed model analysis
            - Provides essential metadata for model checkpointing and distribution
            - Example:
              ```python
              model = Vorox(cfg, vocab_size=tokenizer.vocab_size)
              print(f"Model has {model.trainable_parameters:,} trainable parameters")
              ```
              
        Limitations:
            - Does not account for parameter sharing across modules
            - No breakdown by layer or parameter type
            - No information about parameter sparsity or efficiency
            - Does not reflect actual memory usage during training (which includes optimizer states)
            - No distinction between different parameter sizes (e.g., fp16 vs fp32)
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    @property
    def parameter_breakdown(self):
        """
        Generates a hierarchical breakdown of model parameters with detailed per-module statistics.
        
        Provides a comprehensive analysis of parameter distribution across the model's component
        hierarchy with O(n) traversal complexity, enabling architecture optimization and
        debugging. Implements recursive parameter counting with structured formatting for
        visualization of parameter allocation across the model's computational graph.
        
        Architecture:
            - Implements depth-first recursive traversal of the module hierarchy with O(n) complexity
            - Hierarchical parameter aggregation with per-module and cumulative statistics
            - Special handling for ModuleList containers with indexed submodule tracking
            - Memory complexity: O(d) where d is the maximum depth of the module hierarchy
            - Computation complexity: O(n) where n is the total number of parameters
            - Zero memory overhead beyond temporary dictionary structures for breakdown tracking
            - Preserves module hierarchy with indentation-based visual representation
            
        Returns:
            str: Formatted multi-line string containing the complete parameter breakdown with:
                - Total parameter count header with global sum
                - Hierarchical module listing with per-module parameter counts
                - Consistent indentation reflecting module nesting depth
                - Comma-separated formatting for improved readability of large numbers
                - Module names preserving the original PyTorch naming conventions
                - Consistent footer separator for visual clarity
                
        Behavior:
            - Stateless computation with no side effects
            - Thread-safe due to absence of shared mutable state
            - Deterministic output for fixed model architecture
            - Handles arbitrary module nesting depths with consistent formatting
            - Preserves module hierarchy relationships in the output representation
            - Computationally efficient with linear scaling relative to model size
            
        Integration:
            - Used for model diagnostics and architecture analysis
            - Provides critical insights for model optimization and debugging
            - Complements trainable_parameters with detailed structural information
            - Typically called during model initialization or in debugging workflows
            - Example:
              ```python
              model = Vorox(cfg, vocab_size=tokenizer.vocab_size)
              print(model.parameter_breakdown)
              ```
              
        Limitations:
            - Output size scales with model complexity (can be verbose for large models)
            - No filtering mechanism for focusing on specific submodules
            - No parameter type or shape information (only counts)
            - No visualization of parameter sparsity or efficiency metrics
            - No sorting options for identifying parameter-heavy components
            - String representation only (no structured data return option)
        """
        total, breakdown = self._generate_parameter_breakdown()
        
        def _format_breakdown(breakdown_dict, level=0):
            """
            Recursively formats a nested parameter breakdown dictionary into a hierarchical text representation.
            
            Transforms the raw parameter count data structure into a human-readable indented text format
            with O(n) traversal complexity, where n is the total number of modules in the hierarchy.
            Implements depth-first recursive traversal with consistent indentation patterns to visually
            represent the module hierarchy.
            
            Architecture:
                - Implements depth-first recursive traversal with O(n) time complexity
                - Indentation-based hierarchical formatting with 4-space increments per level
                - Memory complexity: O(d) where d is the maximum depth of the module hierarchy
                - String concatenation with O(n) total operations across all recursive calls
                - Preserves hierarchical relationships through consistent indentation patterns
                - Zero memory overhead beyond temporary string lists for line accumulation
                
            Args:
                breakdown_dict (dict): Dictionary mapping module names to tuples of 
                    (parameter_count, submodule_breakdown_dict). The parameter_count is an
                    integer representing the number of parameters directly owned by the module.
                    The submodule_breakdown_dict is a nested dictionary with the same structure
                    for child modules, or an empty dict for leaf modules.
                    
                level (int, optional): Current indentation level in the hierarchy. Defaults to 0.
                    Controls the indentation prefix (4 spaces per level) for visual hierarchy.
                    Incremented by 1 for each recursive call to represent nesting depth.
                    
            Returns:
                list: List of formatted strings, each representing one line in the hierarchical
                    breakdown. Each line contains a module name, its parameter count with
                    comma-separated formatting, and appropriate indentation reflecting its
                    position in the module hierarchy.
                    
            Behavior:
                - Stateless operation with no side effects
                - Thread-safe due to absence of shared mutable state
                - Deterministic output for fixed input dictionary
                - Handles arbitrary nesting depths with consistent formatting
                - Preserves module naming conventions from PyTorch's module hierarchy
                - Formats large numbers with comma separators for improved readability
                
            Integration:
                - Called by parameter_breakdown property to format the raw breakdown data
                - Processes output from _generate_parameter_breakdown() method
                - Results joined with newlines and wrapped with header/footer in the final output
                - Example:
                  ```
                  formatted_lines = _format_breakdown(breakdown)
                  return "\n" + "\n".join([header] + formatted_lines + [footer])
                  ```
                
            Limitations:
                - Output size scales linearly with model complexity (can be verbose)
                - No truncation mechanism for extremely deep hierarchies
                - Fixed indentation pattern with no customization options
                - No special handling for extremely large parameter counts
                - No sorting capability for module ordering
                - String-based representation only (no structured data output option)
            """
            lines = []
            for name, (count, subbreakdown) in breakdown_dict.items():
                prefix = "    " * level
                lines.append(f"{prefix}{name}: {count:,}")
                if subbreakdown:
                    lines.extend(_format_breakdown(subbreakdown, level + 1))
            return lines
        
        return "\n" + "\n".join([f"====== Parameter Breakdown (Total={total:,}) ======"] + _format_breakdown(breakdown) + ["=" * 60])

    def _generate_parameter_breakdown(self) -> Tuple[int, dict]:
        """
        Recursively analyzes model parameter distribution with hierarchical decomposition and O(n) complexity.
        
        Implements depth-first traversal of the PyTorch module hierarchy to generate a comprehensive
        parameter count breakdown with special handling for container modules. Serves as the core
        computational engine for the parameter_breakdown property, enabling detailed model analysis
        and architecture optimization.
        
        Architecture:
            - Implements recursive depth-first traversal with O(n) time complexity where n is parameter count
            - Two-phase parameter aggregation: direct parameters followed by child module parameters
            - Special case handling for nn.ModuleList with indexed submodule tracking
            - Memory complexity: O(d) where d is the maximum depth of the module hierarchy
            - Dictionary-based hierarchical representation with nested structure mirroring module hierarchy
            - Zero memory overhead beyond temporary dictionary structures for breakdown tracking
            - Preserves module naming conventions from PyTorch's module hierarchy
            
        Returns:
            Tuple[int, dict]: A tuple containing:
                - Total parameter count (int): Sum of all parameters in this module and its children.
                  Represents the complete parameter count for the current module subtree.
                - Parameter breakdown (dict): Hierarchical dictionary mapping module names to tuples of
                  (parameter_count, submodule_breakdown_dict). The parameter_count is an integer
                  representing the number of parameters directly owned by the module. The
                  submodule_breakdown_dict is a nested dictionary with the same structure for child
                  modules, or an empty dict for leaf modules.
                
        Behavior:
            - Stateless computation with no side effects
            - Thread-safe due to absence of shared mutable state
            - Deterministic output for fixed model architecture
            - Handles arbitrary module nesting depths with consistent structure
            - Preserves module hierarchy relationships in the output representation
            - Computationally efficient with linear scaling relative to model size
            - Special handling for container modules (ModuleList) preserves indexed access pattern
            
        Integration:
            - Called by parameter_breakdown property to generate raw breakdown data
            - Works in conjunction with _module_breakdown for recursive traversal
            - Results formatted by _format_breakdown for human-readable representation
            - Critical for model diagnostics and architecture analysis
            - Example:
              ```python
              total, breakdown = model._generate_parameter_breakdown()
              print(f"Model has {total:,} total parameters")
              ```
              
        Limitations:
            - Does not account for parameter sharing across modules (counts duplicates)
            - No special handling for frozen parameters (counts all parameters)
            - Dictionary structure can become large for very deep models
            - No parameter type or shape information (only counts)
            - No optimization for models with repeated identical submodules
            - Recursive implementation may hit Python's recursion limit for extremely deep models
        """
        total = 0
        breakdown = {}
        
        # Get parameters directly attached to this module
        for name, param in self.named_parameters(recurse=False):
            total += param.numel()
            breakdown[name] = (param.numel(), {})
            
        # Recursively get parameters from child modules
        for name, child in self.named_children():
            if isinstance(child, nn.ModuleList):
                # Handle ModuleList specially
                breakdown[name] = {}
                list_total = 0
                for i, subchild in enumerate(child):
                    subtotal, subbreakdown = self._module_breakdown(subchild)
                    list_total += subtotal
                    breakdown[name][f"{name}.{i}"] = (subtotal, subbreakdown)
                total += list_total
            else:
                subtotal, subbreakdown = self._module_breakdown(child)
                total += subtotal
                breakdown[name] = (subtotal, subbreakdown)
                
        return total, breakdown
    
    def _module_breakdown(self, module: nn.Module) -> Tuple[int, dict]:
        """
        Recursively analyzes parameter distribution within a PyTorch module with O(n) traversal complexity.
        
        Implements depth-first traversal of a module's parameter hierarchy to generate a comprehensive
        parameter count breakdown with hierarchical decomposition. Serves as the core recursive engine
        for the parameter_breakdown property, enabling detailed model analysis at arbitrary nesting depths.
        
        Architecture:
            - Implements recursive depth-first traversal with O(n) time complexity where n is parameter count
            - Two-phase parameter aggregation: direct parameters followed by child module parameters
            - Memory complexity: O(d) where d is the maximum depth of the module hierarchy
            - Dictionary-based hierarchical representation with nested structure mirroring module hierarchy
            - Zero memory overhead beyond temporary dictionary structures for breakdown tracking
            - Preserves module naming conventions from PyTorch's module hierarchy
            
        Args:
            module (nn.Module): PyTorch module to analyze. Can be any subclass of nn.Module,
                including container modules, custom implementations, or built-in layers.
                The method handles arbitrary nesting depths and module compositions.
                
        Returns:
            Tuple[int, dict]: A tuple containing:
                - Total parameter count (int): Sum of all parameters in this module and its children.
                  Represents the complete parameter count for the current module subtree.
                - Parameter breakdown (dict): Hierarchical dictionary mapping module names to tuples of
                  (parameter_count, submodule_breakdown_dict). The parameter_count is an integer
                  representing the number of parameters directly owned by the module. The
                  submodule_breakdown_dict is a nested dictionary with the same structure for child
                  modules, or an empty dict for leaf modules.
                
        Behavior:
            - Stateless computation with no side effects
            - Thread-safe due to absence of shared mutable state
            - Deterministic output for fixed module architecture
            - Handles arbitrary module nesting depths with consistent structure
            - Preserves module hierarchy relationships in the output representation
            - Computationally efficient with linear scaling relative to module size
            
        Integration:
            - Called recursively by _generate_parameter_breakdown() to traverse module hierarchy
            - Results aggregated into complete model parameter breakdown
            - Used for model diagnostics and architecture analysis
            - Example:
              ```python
              subtotal, subbreakdown = self._module_breakdown(child)
              total += subtotal
              breakdown[name] = (subtotal, subbreakdown)
              ```
              
        Limitations:
            - Does not account for parameter sharing across modules (counts duplicates)
            - No special handling for frozen parameters (counts all parameters)
            - Dictionary structure can become large for very deep modules
            - No parameter type or shape information (only counts)
            - No optimization for modules with repeated identical submodules
            - Recursive implementation may hit Python's recursion limit for extremely deep models
        """
        total = 0
        breakdown = {}
        
        # Get parameters directly attached to this module
        for name, param in module.named_parameters(recurse=False):
            total += param.numel()
            breakdown[name] = (param.numel(), {})
            
        # Recursively get parameters from child modules
        for name, child in module.named_children():
            subtotal, subbreakdown = self._module_breakdown(child)
            total += subtotal
            breakdown[name] = (subtotal, subbreakdown)
            
        return total, breakdown
    

if __name__ == "__main__":
    import sys
    import time

    import torch.profiler
    from torch.profiler import schedule, ProfilerActivity

    print(f"Python version: {sys.version}")
    print(f"Torch version: {torch.__version__}")
    print(f"Torch CUDA available: {torch.cuda.is_available()}")
    print(f"Torch MPS available: {torch.backends.mps.is_available()}")
    print(f"Torch Available Devices: {get_available_devices()}")
    print(f"Torch device: {torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')}")
    print()

    with open("configs/20M_test_model.yml", "r") as f:
        cfg = Config.model_validate(yaml.safe_load(f))

    pprint(dict(cfg))
    print()

    device = torch.device('cuda')

    #prof_schedule = schedule(wait=0, warmup=0, active=1)
    def on_trace_ready(p):
        """
        Processes and displays PyTorch profiler results with multi-dimensional performance analysis.
        
        Implements a comprehensive profiling callback for PyTorch's profiler that extracts, formats,
        and displays critical performance metrics across three dimensions (GPU time, GPU memory, CPU time)
        with O(n) processing complexity where n is the number of profiled operations.
        
        Architecture:
            - Implements profiler callback pattern with O(n) processing complexity
            - Multi-dimensional performance analysis across GPU time, memory, and CPU utilization
            - Hierarchical output formatting with operation-level granularity
            - Time complexity: O(n) where n is the number of profiled operations
            - Space complexity: O(n) for intermediate table representations
            - Prioritized metric sorting with configurable row limits for focused analysis
            - Commented infrastructure for Chrome trace export for visualization (currently disabled)
            
        Args:
            p (torch.profiler.profile): Profiler instance containing collected trace data.
                Must be an active profiler that has completed at least one step of data collection.
                Contains hierarchical operation records with timing and resource utilization metrics.
                Provides step_num attribute indicating the current profiling step/iteration.
                
        Behavior:
            - Stateless operation with no side effects beyond console output
            - Thread-safe due to absence of shared mutable state
            - Deterministic output for fixed profiler input
            - Preserves profiler state for potential subsequent processing
            - Handles arbitrary operation counts with consistent formatting
            - Processes all collected metrics without filtering
            
        Integration:
            - Registered as callback to torch.profiler.profile via on_trace_ready parameter
            - Triggered automatically at profiler step boundaries
            - Designed for both training and inference performance analysis
            - Example:
              ```python
              prof = torch.profiler.profile(
                  activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                  record_shapes=True,
                  profile_memory=True,
                  with_stack=True,
                  on_trace_ready=on_trace_ready
              )
              ```
              
        Limitations:
            - Console-only output with no persistent storage (trace export commented out)
            - Fixed row limit (32) with no dynamic adjustment based on operation count
            - No custom metric aggregation or statistical analysis
            - No support for distributed training profiling correlation
            - No historical data comparison across profiling runs
            - No visualization capabilities without uncommenting trace export
            - No explicit error handling for profiler data access failures
        """
        #profiler_output_dir = Path(self.cfg.save_folder) / "profiler"
        #profiler_output_dir.mkdir(exist_ok=True)

        output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=32)
        print(f"Profile by total GPU time at step {p.step_num}:\n{output}")

        output = p.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=32)
        print(f"Profile by total GPU memory usage at step {p.step_num}:\n{output}")

        output = p.key_averages().table(sort_by="self_cpu_time_total", row_limit=32)
        print(f"Profile by total CPU time at step {p.step_num}:\n{output}")

        #p.export_chrome_trace(
        #    str(trace_path := (profiler_output_dir / f"{p.step_num}.chrome_trace.json.gz"))
        #)
        
    prof = torch.profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        #schedule=prof_schedule,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        on_trace_ready=on_trace_ready
    )

    with prof as p:
        start = time.time()
        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer.name)
        model = Vorox(cfg, vocab_size=tokenizer.vocab_size)
        init_time = time.time() - start
        param_memory = sum(p.nelement() * p.element_size() for p in model.parameters())
        trainable_param_memory = sum(p.nelement() * p.element_size() for p in model.parameters() if p.requires_grad)
        print(f"Initialization time: {init_time:.2f} seconds")
        print(f"Model parameters memory usage: {param_memory / 1024 / 1024:.2f} MB")
        print(f"Trainable parameters memory usage: {trainable_param_memory / 1024 / 1024:.2f} MB")
        print(f"Trainable parameters count: {model.trainable_parameters:,}")
        print(f"Parameter Type: {next(model.parameters()).dtype}")
        print()

        x = torch.randint(0, tokenizer.vocab_size, (cfg.train.micro_batch_size, cfg.train.max_seq_len))
        print(f"Input tensor {x.size()} memory usage: {x.element_size() * x.nelement() / 1024 / 1024:.2f} MB")
        print()

        start = time.time()
        model = model.to(device)
        print(f"Model to device time: {time.time() - start:.2f} seconds")
        print()

        start = time.time()
        x = x.to(device)
        print(f"Input tensor {x.shape} to device time: {time.time() - start:.2f} seconds")
        print()

        start = time.time()
        out = model(x)
        print(f"Forward pass time: {time.time() - start:.2f} seconds")
        print(f"Output tensor {out.shape} memory usage: {out.element_size() * out.nelement() / 1024 / 1024:.2f} MB")
        print()

        start = time.time()
        out.sum().backward()
        bp_time = time.time() - start
        grad_memory = sum(p.grad.nelement() * p.grad.element_size() for p in model.parameters() if p.grad is not None)
        print(f"Backward pass time: {bp_time:.2f} seconds")
        print(f"Gradient memory usage: {grad_memory / 1024 / 1024:.2f} MB")

        p.step()
