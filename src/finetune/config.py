"""Shared configuration models for training runs."""

from enum import StrEnum
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class TrainingMethod(StrEnum):
    FULL = "full"
    LORA = "lora"
    QLORA = "qlora"


class BaseTrainingConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FT_", env_file=".env", extra="ignore")

    model_name_or_path: str = Field(description="HuggingFace model ID or local path")
    output_dir: Path = Field(default=Path("outputs"), description="Directory for checkpoints and logs")
    dataset_name: str = Field(description="HuggingFace dataset ID or local path")
    dataset_split: str = "train"
    max_seq_length: int = 2048

    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.01

    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500
    save_total_limit: int = 3

    bf16: bool = True
    tf32: bool = True
    gradient_checkpointing: bool = True

    seed: int = 42
    report_to: str = "wandb"
    run_name: str = "finetune-run"


class LoRAConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LORA_", env_file=".env", extra="ignore")

    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: list[str] = Field(default_factory=lambda: ["q_proj", "v_proj"])


class QLoRAConfig(LoRAConfig):
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
