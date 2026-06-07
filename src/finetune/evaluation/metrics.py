"""Evaluation metrics: perplexity and qualitative generation samples."""

import math

import torch
from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase


@torch.inference_mode()
def compute_perplexity(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset: Dataset,
    text_column: str = "text",
    max_length: int = 2048,
    stride: int = 512,
) -> float:
    """Compute sliding-window perplexity over a dataset.

    Args:
        model: Fine-tuned causal LM in eval mode.
        tokenizer: Matching tokenizer.
        dataset: Dataset with a text column.
        text_column: Name of the text field.
        max_length: Context window length.
        stride: Step between windows; smaller values are more accurate but slower.

    Returns:
        Perplexity score (lower is better).
    """
    model.eval()
    device = next(model.parameters()).device
    total_nll = 0.0
    total_tokens = 0

    for example in dataset:
        encodings = tokenizer(example[text_column], return_tensors="pt")
        seq_len = encodings.input_ids.size(1)

        prev_end = 0
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            outputs = model(input_ids, labels=target_ids)
            total_nll += outputs.loss.item() * trg_len
            total_tokens += trg_len
            prev_end = end_loc
            if end_loc == seq_len:
                break

    return math.exp(total_nll / total_tokens)


@torch.inference_mode()
def generate_samples(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> list[str]:
    """Generate text completions for a list of prompts.

    Args:
        model: Fine-tuned causal LM in eval mode.
        tokenizer: Matching tokenizer.
        prompts: Input prompt strings.
        max_new_tokens: Maximum tokens to generate per prompt.
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.

    Returns:
        List of generated strings (prompt + completion).
    """
    model.eval()
    device = next(model.parameters()).device
    results: list[str] = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        results.append(tokenizer.decode(output_ids[0], skip_special_tokens=True))

    return results
