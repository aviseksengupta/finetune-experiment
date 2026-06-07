"""Model and tokenizer loading for full, LoRA, and QLoRA training."""

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from finetune.config import LoRAConfig, QLoRAConfig, TrainingMethod


def load_model_and_tokenizer(
    model_name_or_path: str,
    method: TrainingMethod = TrainingMethod.FULL,
    lora_config: LoRAConfig | None = None,
    qlora_config: QLoRAConfig | None = None,
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map: str | dict = "auto",
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a causal LM with the appropriate precision and adapter configuration.

    Args:
        model_name_or_path: HuggingFace model ID or local path.
        method: Training technique — full, lora, or qlora.
        lora_config: LoRA hyperparameters (required for LORA method).
        qlora_config: QLoRA hyperparameters (required for QLORA method).
        torch_dtype: Compute dtype for full/LoRA training.
        device_map: Device placement strategy passed to from_pretrained.

    Returns:
        Tuple of (model, tokenizer).
    """
    tokenizer = _load_tokenizer(model_name_or_path)

    if method == TrainingMethod.QLORA:
        model = _load_qlora_model(model_name_or_path, qlora_config or QLoRAConfig(), device_map)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )

    if method in (TrainingMethod.LORA, TrainingMethod.QLORA):
        cfg = lora_config or LoRAConfig()
        peft_config = LoraConfig(
            r=cfg.r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias=cfg.bias,
            task_type=TaskType.CAUSAL_LM,
            target_modules=cfg.target_modules,
        )
        if method == TrainingMethod.QLORA:
            model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    return model, tokenizer


def _load_tokenizer(model_name_or_path: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_qlora_model(
    model_name_or_path: str,
    cfg: QLoRAConfig,
    device_map: str | dict,
) -> PreTrainedModel:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg.load_in_4bit,
        bnb_4bit_compute_dtype=getattr(torch, cfg.bnb_4bit_compute_dtype),
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
    )
    return AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=bnb_config,
        device_map=device_map,
    )
