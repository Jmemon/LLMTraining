from enum import Enum
from typing import Optional, Literal, List, Union
from pathlib import Path
from pydantic import BaseModel
import yaml




class ActivationType(str, Enum):
    gelu = "gelu"
    relu = "relu"
    silu = "silu"
    swiglu = "swiglu"

class ModelConfig(BaseModel):
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

class OptimizerConfig(BaseModel):
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

class LossConfig(BaseModel):
    name: LossType

class TrainConfig(BaseModel):
    epochs: int

class DatasetType(str, Enum):
    dclm_baseline = "dclm_baseline"
    # thestack = "thestack"
    # dolma = "dolma"
    # redpajama = "redpajama"

class EvaluatorType(str, Enum):
    mmlu = "mmlu"
    gsm8k = "gsm8k"
    gsm_symbolic = "gsm_symbolic"

class EvaluatorConfig(BaseModel):
    evaluators: List[EvaluatorType]
    batch_size: int
    num_workers: int
    prefetch_size: int

class DataConfig(BaseModel):
    prefetch_size: int
    shuffle_buffer: bool = False
    num_workers: int = 4
    macro_batch_size: int
    micro_batch_size: int
    max_seq_len: int
    train_data: Union[List[DatasetType], None] = None
    # Additional settings (e.g. timeouts) can be added here

class MetricsConfig(BaseModel):
    compute_metrics: bool
    train_metrics: List[Literal["loss"]]
    val_metrics: List[Literal["loss"]]
    val_check_interval: float = 1.0
    early_stopping: bool
    early_stopping_patience: int
    early_stopping_min_delta: float

class LoggingConfig(BaseModel):
    wandb_project: str
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_tags: List[str] = []
    log_every_n_steps: int = 50

class CheckpointConfig(BaseModel):
    save_top_k: int = 3
    checkpoint_dir: str
    monitor: str = "val/loss"
    mode: Literal["min", "max"] = "min"
    save_last: bool = True
    save_every_n_steps: int = 1000
    load_from_checkpoint: Optional[str] = None

class HardwareConfig(BaseModel):
    device: Literal["cpu", "cuda", "mps"]
    precision: Literal["fp32", "fp16", "bf16"]
    distributed: bool = False
    num_gpus: int = 1

class RunConfig(BaseModel):
    experiment_name: str
    seed: int
    model: ModelConfig
    optimizer: OptimizerConfig
    loss: LossConfig
    train: TrainConfig
    data: DataConfig
    hardware: HardwareConfig
    metrics: MetricsConfig
    logging: LoggingConfig
    checkpoint: CheckpointConfig
    eval: EvaluatorConfig

if __name__ == "__main__":
    with open("configs/20M_test_model.yml", "r") as f:
        cfg = RunConfig.model_validate(yaml.safe_load(f))

    print(cfg)
