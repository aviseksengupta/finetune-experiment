# finetune-experiment

LLM fine-tuning experiments using full training, LoRA, and QLoRA.

## Project layout

```
src/finetune/
  config.py          Pydantic settings for training, LoRA, QLoRA configs
  cli.py             Typer CLI (finetune train / finetune evaluate)
  data/loader.py     Dataset loading + Alpaca/chat format helpers
  models/loader.py   Model + tokenizer loading for all three methods
  trainers/sft.py    SFTTrainer wrapper (TRL)
  evaluation/        Perplexity + generation sample helpers
  utils/helpers.py   Seed, device, Rich logging

configs/             TOML experiment configs (not loaded automatically)
notebooks/           Exploration notebooks
scripts/             One-off helper scripts
tests/               pytest suite
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

For flash-attention (CUDA only):
```bash
pip install -e ".[dev,flash-attn]"
```

## Running experiments

```bash
# LoRA fine-tune
finetune train --model mistralai/Mistral-7B-v0.1 --dataset tatsu-lab/alpaca --method lora

# QLoRA (lower VRAM)
finetune train --model meta-llama/Llama-2-7b-hf --dataset tatsu-lab/alpaca --method qlora

# Evaluate a checkpoint
finetune evaluate --model outputs/mistral-7b-alpaca-lora --dataset tatsu-lab/alpaca \
  --prompt "What is the capital of France?"
```

## Tests

```bash
pytest
```

## Linting / type checking

```bash
ruff check src tests
mypy src
```

## Key dependencies

| Package | Purpose |
|---------|---------|
| transformers | Model loading, tokenization |
| peft | LoRA / QLoRA adapters |
| bitsandbytes | 4-bit quantization for QLoRA |
| trl | SFTTrainer, reward model helpers |
| accelerate | Multi-GPU / DeepSpeed distribution |
| datasets | HuggingFace dataset hub |
| wandb | Experiment tracking |

## Environment variables

Copy `.env.example` to `.env` and fill in:

```
FT_MODEL_NAME_OR_PATH=
FT_DATASET_NAME=
WANDB_API_KEY=
HF_TOKEN=
```
