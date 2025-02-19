from enum import Enum
from pydantic import BaseModel
import yaml


class Tokenizer(BaseModel):
    name: str

class ActivationType(str, Enum):
    gelu = "gelu"
    relu = "relu"
    silu = "silu"
    swiglu = "swiglu"

class Architecture(BaseModel):
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

class LossType(str, Enum):
    mse = "mse"
    cross_entropy = "cross_entropy"
    perplexity = "perplexity"

class Loss(BaseModel):
    type: LossType

class Train(BaseModel):
    epochs: int
    macro_batch_size: int
    micro_batch_size: int
    max_seq_len: int

class Dataset(str, Enum):
    dclm_baseline = "dclm_baseline"
    thestack = "thestack"
    dolma = "dolma"
    redpajama = "redpajama"

class DataSettings(BaseModel):
    prefetch_size: int
    cache_dsn: str  # PostgreSQL DSN (e.g., postgresql://user:pass@host:port/db)
    shuffle_buffer: bool = False
    num_workers: int = 4
    # Additional settings (e.g. timeouts) can be added here

class TrainingDataConfig(BaseModel):
    settings: DataSettings
    urls: list[str]

class Device(str, Enum):
    cpu = "cpu"
    cuda = "cuda"
    mps = "mps"

class Config(BaseModel):
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
