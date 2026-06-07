"""Dataset loading with preprocessing for instruction/chat formats."""

from collections.abc import Callable
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset
from transformers import PreTrainedTokenizerBase


def load_dataset_for_training(
    dataset_name: str,
    split: str = "train",
    tokenizer: PreTrainedTokenizerBase | None = None,
    format_fn: Callable[[dict[str, Any]], str] | None = None,
    max_seq_length: int = 2048,
    num_proc: int = 4,
) -> Dataset | DatasetDict:
    """Load and optionally tokenize a dataset for fine-tuning.

    Args:
        dataset_name: HuggingFace dataset ID or local path.
        split: Dataset split to load.
        tokenizer: If provided, tokenizes the formatted text.
        format_fn: Maps a dataset example to a single training string.
        max_seq_length: Truncation length when tokenizing.
        num_proc: Parallel workers for map operations.

    Returns:
        Raw or tokenized dataset ready for a Trainer.
    """
    dataset = load_dataset(dataset_name, split=split)

    if format_fn is not None:
        dataset = dataset.map(
            lambda example: {"text": format_fn(example)},
            num_proc=num_proc,
            remove_columns=dataset.column_names,
        )

    if tokenizer is not None:
        dataset = dataset.map(
            lambda batch: tokenizer(
                batch["text"],
                truncation=True,
                max_length=max_seq_length,
                padding=False,
            ),
            batched=True,
            num_proc=num_proc,
            remove_columns=["text"],
        )

    return dataset


def alpaca_format(example: dict[str, Any]) -> str:
    """Format an Alpaca-style example into a prompt string."""
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")

    if input_text:
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
    return prompt


def chat_format(example: dict[str, Any], tokenizer: PreTrainedTokenizerBase) -> str:
    """Format a chat-style example using the tokenizer's chat template."""
    messages = example.get("messages", example.get("conversations", []))
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
