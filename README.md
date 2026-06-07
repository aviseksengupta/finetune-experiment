# finetune-experiment

Experimental framework for fine-tuning large language models using full training, LoRA, and QLoRA techniques.

## Goals

- Compare full fine-tuning vs. parameter-efficient methods (LoRA, QLoRA) on the same tasks
- Validate results with perplexity and qualitative generation samples
- Keep experiments reproducible via config files and W&B tracking

## Techniques covered

| Method | Description | VRAM requirement |
|--------|-------------|-----------------|
| Full fine-tuning | All parameters updated | High (A100 80GB+) |
| LoRA | Low-rank adapter matrices on attention layers | Medium (24–40 GB) |
| QLoRA | LoRA + 4-bit NF4 quantization of base model | Low (12–16 GB) |

## Quick start

### 1. Clone and set up

```bash
git clone https://github.com/aviseksengupta/finetune-experiment
cd finetune-experiment

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### 2. Configure

```bash
cp .env.example .env
# edit .env — add WANDB_API_KEY and HF_TOKEN
```

### 3. Train

```bash
# LoRA on Mistral-7B with the Alpaca dataset
finetune train \
  --model mistralai/Mistral-7B-v0.1 \
  --dataset tatsu-lab/alpaca \
  --method lora \
  --output-dir outputs/mistral-alpaca-lora \
  --run-name mistral-alpaca-lora

# QLoRA on LLaMA-2-7B (lower VRAM)
finetune train \
  --model meta-llama/Llama-2-7b-hf \
  --dataset tatsu-lab/alpaca \
  --method qlora
```

### 4. Evaluate

```bash
finetune evaluate \
  --model outputs/mistral-alpaca-lora \
  --dataset tatsu-lab/alpaca \
  --prompt "Explain quantum entanglement in simple terms."
```

## Project structure

```
src/finetune/       Core library
  config.py         Pydantic-settings config models
  cli.py            CLI entry point (typer)
  data/             Dataset loading & formatting
  models/           Model/tokenizer loading per method
  trainers/         SFTTrainer wrapper
  evaluation/       Perplexity + generation eval
  utils/            Seed, device, logging helpers

configs/            TOML experiment configuration files
notebooks/          Jupyter exploration notebooks
scripts/            Utility scripts
tests/              pytest test suite
```

## Development

```bash
pytest                  # run tests
ruff check src tests    # lint
mypy src                # type check
```

## Requirements

- Python 3.11+
- CUDA GPU recommended for training (MPS supported for local dev/inference)
- [Weights & Biases](https://wandb.ai) account for experiment tracking
- [HuggingFace](https://huggingface.co) account for gated models (LLaMA etc.)
