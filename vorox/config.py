from enum import Enum
from typing import Optional, Literal, List
from pydantic import BaseModel
import yaml



class ActivationType(str, Enum):
    gelu = "gelu"
    relu = "relu"
    silu = "silu"
    swiglu = "swiglu"

class Model(BaseModel):
    tokenizer_name: str
    n_layers: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    hidden_size: int
    activation: ActivationType
    rope: bool
    rope_theta: int

class OptimizerType(str, Enum):
    adamw = "adamw"
    adam = "adam"
    sgd = "sgd"

class Optimizer(BaseModel):
    type: OptimizerType
    lr: float
    betas: list[float]
    weight_decay: float
    momentum: float = 0.0
    warmup_steps: int = 0
    warmup_ratio: float = 0.0
    scheduler: Optional[Literal["cosine", "constant"]] = None
    gradient_clip_val: float = 1.0

class LossType(str, Enum):
    mse = "mse"
    cross_entropy = "cross_entropy"
    perplexity = "perplexity"

class Loss(BaseModel):
    name: LossType

class Train(BaseModel):
    epochs: int

class Dataset(str, Enum):
    dclm_baseline = "dclm_baseline"
    thestack = "thestack"
    dolma = "dolma"
    redpajama = "redpajama"

class Data(BaseModel):
    prefetch_size: int
    cache_dsn: str  # PostgreSQL DSN (e.g., postgresql://user:pass@host:port/db)
    shuffle_buffer: bool = False
    num_workers: int = 4
    macro_batch_size: int
    micro_batch_size: int
    max_seq_len: int
    train_data: List[Dataset]
    val_data: List[Dataset]
    # Additional settings (e.g. timeouts) can be added here

class Hardware(BaseModel):
    device: Literal["cpu", "cuda", "mps"]
    precision: Literal["fp32", "fp16", "bf16"]
    distributed: bool = False
    num_gpus: int = 1

class Config(BaseModel):
    model: Model
    optimizer: Optimizer
    loss: Loss
    train: Train
    data: Data
    hardware: Hardware

if __name__ == "__main__":
    with open("configs/20M_test_model.yml", "r") as f:
        cfg = Config.model_validate(yaml.safe_load(f))

    print(cfg)
