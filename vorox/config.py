
from pydantic import BaseModel


class Tokenizer(BaseModel):
    name: str

class Architecture(BaseModel):
    n_layers: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    rope: bool
    rope_theta: int

class Train(BaseModel):
    epochs: int
    batch_size: int
    learning_rate: float
    max_seq_len: int

class Config(BaseModel):
    tokenizer: Tokenizer
    architecture: Architecture
    train: Train
