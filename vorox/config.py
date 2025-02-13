from enum import Enum
from pydantic import BaseModel
import yaml


class Tokenizer(BaseModel):
    name: str

class Architecture(BaseModel):
    n_layers: int
    d_model: int
    n_heads: int
    n_kv_heads: int
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
    batch_size: int
    max_seq_len: int

class Config(BaseModel):
    tokenizer: Tokenizer
    architecture: Architecture
    optimizer: Optimizer
    loss: Loss
    train: Train

if __name__ == "__main__":
    with open("configs/20M_test_model.yml", "r") as f:
        cfg = Config.model_validate(yaml.safe_load(f))

    print(cfg)
