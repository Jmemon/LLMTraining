from typing import Iterable, Tuple
import yaml

from pprint import pprint
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from vorox.config import Config
from vorox.utils import get_available_devices

"""
What's left?
implement rope.
"""


class RoPE(nn.Module):
    """
    Rotary Position Embedding.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

    def get_rotary_embeddings(self, seq_len: int):
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
        B, n_heads, T, head_size = x.size()
        x = x.view(B, n_heads, T, 2, head_size // 2)
        x1, x2 = x.unbind(dim=-2)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, pos_sin: torch.Tensor, pos_cos: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
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
    AdaptiveNorm.
    https://arxiv.org/pdf/1911.07013

    LayerNorm(x):
        h = dot(g, z(x)) + b
    where:
        z(x) = (x - mean(x)) / sqrt(var(x) + eps) – Ie take mean and stdev across the sample dimension, then normalize sample with them 
        g is the gain (same shape as x)
        b is the bias (same shape as x)
        If we assume the distribution over the elements of a sample is normal, then z(x) is normalized version to N(0, 1), then g and b rescale it to N(b, g).
        Both g and b are learned parameters.

    In the above paper it was shown that the bias and gain actually don't improve performance that much, and actually cause overfitting. So they introduced AdaptiveNorm(x):
        h = dot(phi(z(x)), z(x))
    where:
        z(x) is the same as in LayerNorm.
        phi is an elementwise function to scale each element of z(x) based on its value. They derive a function.
    
    Their results weren't that dramatically improved, so I'm sticking with LayerNorm. Although whether or not to include bias/gain is something for me to determine.
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
        d_model: int, the dimension of the input (ie the embedding dimension)
        n_heads: int, the number of attention heads (also the number of queries)
        n_kv_heads: int, the number of key/value heads (ie the number of groups)
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

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: torch.Tensor, the input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            torch.Tensor, the output tensor of shape (batch_size, seq_len, d_model)
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
        attn = attn.softmax(dim=-1) @ v  # shape (B, n_heads, L, d_v)
        attn = attn.transpose(1, 2).reshape(B, L, self.n_heads * self.d_v)  # shape (B, L, n_heads * d_v)

        out = self.out_proj(attn)  # shape (B, L, d_model)
        return out


class VoroxDecoderBlock(nn.Module):
    """
    Decoder-only transformer block.
        ``(MLP∘LN-simple)((Attn∘LN-simple)(x) + x) + interm_x``
    """

    def __init__(self, cfg: Config):
        """
        Args:
            dim: int, the dimension of the input
            n_heads: int, the number of attention heads
        """
        super().__init__()

        assert cfg.architecture.n_heads % 4 == 0, "n_heads must be divisible by 4"

        self.d_model = cfg.architecture.d_model
        self.n_heads = cfg.architecture.n_heads
        self.n_kv_heads = cfg.architecture.n_heads // 4

        self.attn_norm = LayerNorm((self.d_model,), elementwise_affine=False)
        self.attn = GroupedQueryAttention(cfg)
        self.mlp_norm = LayerNorm((self.d_model,), elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 4),
            nn.GELU(),
            nn.Linear(self.d_model * 4, self.d_model),
        )

    def forward(self, x):
        interm_x = x + self.attn(self.attn_norm(x))
        out_x = interm_x + self.mlp(self.mlp_norm(interm_x))
        return out_x


class Vorox(nn.Module):

    def __init__(self, cfg: Config, vocab_size: int):
        super().__init__()

        self.n_layers = cfg.architecture.n_layers
        self.d_model = cfg.architecture.d_model
        self.n_heads = cfg.architecture.n_heads
        self.vocab_size = vocab_size

        self.emb = nn.Embedding(vocab_size, self.d_model)
        self.transformer_blocks = nn.Sequential(*[VoroxDecoderBlock(cfg) for _ in range(cfg.architecture.n_layers)])
        self.ff_out = nn.Linear(self.d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor):
        x = self.emb(input_ids)
        x = self.transformer_blocks(x)
        x = self.ff_out(x)
        return x
    
    @property
    def trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    @property
    def parameter_breakdown(self):
        """
        Returns a formatted string showing the parameter breakdown hierarchy.
        
        Args:
            indent: Current indentation level (used recursively)
            
        Returns:
            Formatted string showing parameter counts for all modules and submodules
        """
        total, breakdown = self._generate_parameter_breakdown()
        
        def _format_breakdown(breakdown_dict, level=0):
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
        Recursively calculates parameter counts for each module and submodule.
        
        Returns:
            Tuple containing:
            - Total parameter count for this module
            - Dict mapping submodule names to their (count, breakdown) tuples
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
        """Helper function for parameter_breakdown()"""
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

    device = torch.device('mps')

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
    print(model.parameter_breakdown)
    print()

    x = torch.randint(0, tokenizer.vocab_size, (cfg.train.batch_size, cfg.train.max_seq_len))
    print(f"Input tensor {x.size()} memory usage: {x.element_size() * x.nelement() / 1024 / 1024:.2f} MB")
    print()

    start = time.time()
    model = model.to(device)
    print(f"Model to device time: {time.time() - start:.2f} seconds")
    print()

    start = time.time()
    x = x.to(device)
    print(f"Input tensor to device time: {time.time() - start:.2f} seconds")
    print()

    start = time.time()
    out = model(x)
    print(f"Forward pass time: {time.time() - start:.2f} seconds")
    print(f"Output tensor (1, 1024, 512) memory usage: {out.element_size() * out.nelement() / 1024 / 1024:.2f} MB")
    print()

    start = time.time()
    out.sum().backward()
    bp_time = time.time() - start
    grad_memory = sum(p.grad.nelement() * p.grad.element_size() for p in model.parameters() if p.grad is not None)
    print(f"Backward pass time: {bp_time:.2f} seconds")
    print(f"Gradient memory usage: {grad_memory / 1024 / 1024:.2f} MB")
