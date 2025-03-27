from enum import Enum
from typing import Optional, Literal, List, Union
from pathlib import Path
from pydantic import BaseModel, field_validator
import yaml
import torch




class ActivationType(str, Enum):
    gelu = "gelu"
    relu = "relu"
    silu = "silu"
    swiglu = "swiglu"

class ArchitectureConfig(BaseModel):
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
    perplexity = "perplexity"

class LossConfig(BaseModel):
    type: LossType


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
    epochs: int
    epoch_tokens: int
    train_data: Union[List[DatasetType], None] = None
    # Additional settings (e.g. timeouts) can be added here

class MetricsConfig(BaseModel):
    compute_metrics: bool
    train_metrics: List[Literal["loss"]]

class LoggingConfig(BaseModel):
    wandb_project: str
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_tags: List[str] = []
    log_every_n_steps: int = 50

class CheckpointConfig(BaseModel):
    save_top_k: int = 3
    checkpoint_dir: str
    monitor: Optional[EvaluatorType] = None
    mode: Literal["min", "max"] = "max"
    save_last: bool = True
    save_every_n_steps: int = 200
    load_from_checkpoint: Optional[str] = None

class HardwareConfig(BaseModel):
    device: Literal["cpu", "cuda", "mps"]
    precision: Literal["fp32", "fp16", "bf16"]
    distributed: bool = False
    num_gpus: int = 1
    
    @field_validator("device")
    def device_exists(cls, device):
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA device requested but not available")
        elif device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS device requested but not available")
        return device
    
    @field_validator("precision")
    def valid_precision_device_combo(cls, field_value, info):
        device = info.data.get("device")
        precision = field_value
        
        if device == "cpu" and precision in ["fp16", "bf16"]:
            raise ValueError(f"Precision {precision} not supported on CPU. Use fp32 instead.")
        
        if device == "mps" and precision == "bf16":
            raise ValueError("BF16 precision not supported on MPS. Use fp16 or fp32 instead.")
        
        if device == "cuda":
            if precision == "bf16" and not torch.cuda.is_bf16_supported():
                raise ValueError("BF16 precision requested but not supported on this CUDA device")
        
        return precision

class RunConfig(BaseModel):
    experiment_name: str
    seed: int
    architecture: ArchitectureConfig
    optimizer: OptimizerConfig
    loss: LossConfig
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
