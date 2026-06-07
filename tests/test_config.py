"""Smoke tests for configuration models."""

from pathlib import Path

import pytest

from finetune.config import BaseTrainingConfig, LoRAConfig, QLoRAConfig, TrainingMethod


def test_base_training_config_defaults() -> None:
    config = BaseTrainingConfig(
        model_name_or_path="gpt2",
        dataset_name="tatsu-lab/alpaca",
    )
    assert config.num_train_epochs == 3
    assert config.output_dir == Path("outputs")
    assert config.bf16 is True


def test_lora_config_defaults() -> None:
    cfg = LoRAConfig()
    assert cfg.r == 16
    assert cfg.lora_alpha == 32
    assert "q_proj" in cfg.target_modules


def test_qlora_extends_lora() -> None:
    cfg = QLoRAConfig()
    assert cfg.load_in_4bit is True
    assert cfg.bnb_4bit_quant_type == "nf4"


def test_training_method_enum() -> None:
    assert TrainingMethod("lora") == TrainingMethod.LORA
    assert TrainingMethod("qlora") == TrainingMethod.QLORA
    assert TrainingMethod("full") == TrainingMethod.FULL
