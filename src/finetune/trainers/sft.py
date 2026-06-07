"""Supervised fine-tuning (SFT) trainer wrapping TRL's SFTTrainer."""

from pathlib import Path

from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase, TrainingArguments
from trl import SFTTrainer

from finetune.config import BaseTrainingConfig


def run_sft(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: Dataset,
    eval_dataset: Dataset | None,
    config: BaseTrainingConfig,
    extra_training_args: dict | None = None,
) -> Path:
    """Run supervised fine-tuning and return the final checkpoint path.

    Args:
        model: The (possibly PEFT-wrapped) causal LM.
        tokenizer: Matching tokenizer.
        train_dataset: Tokenized training examples.
        eval_dataset: Optional evaluation set.
        config: Shared training hyper-parameters.
        extra_training_args: Any TrainingArguments fields not covered by config.

    Returns:
        Path to the final saved checkpoint.
    """
    training_args = TrainingArguments(
        output_dir=str(config.output_dir),
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        weight_decay=config.weight_decay,
        bf16=config.bf16,
        tf32=config.tf32,
        gradient_checkpointing=config.gradient_checkpointing,
        logging_steps=config.logging_steps,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=config.eval_steps if eval_dataset is not None else None,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        seed=config.seed,
        report_to=config.report_to,
        run_name=config.run_name,
        **(extra_training_args or {}),
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(config.output_dir)

    return config.output_dir
