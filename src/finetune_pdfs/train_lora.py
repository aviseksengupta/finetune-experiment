"""LoRA fine-tuning utilities for causal language models."""

from __future__ import annotations

import tomllib
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console

app = typer.Typer(add_completion=False)
console = Console()


def load_config(config_path: Path) -> dict[str, Any]:
    """Load a TOML training config."""

    with open(config_path, "rb") as f:
        return tomllib.load(f)


def train_lora(
    chunks: Iterable[str],
    cfg: dict[str, Any],
    *,
    dry_run: bool = False,
) -> None:
    """Fine-tune the configured model on pre-chunked text using LoRA."""

    chunk_list = [chunk for chunk in chunks if chunk.strip()]
    if not chunk_list:
        raise ValueError("No non-empty chunks were provided for training.")

    if dry_run:
        console.print(f"[bold]Loaded {len(chunk_list)} training sample(s)[/bold]")
        console.print("\n[yellow]Dry run: stopping before model load/training.[/yellow]")
        console.print(f"Sample chunk:\n\n{chunk_list[0][:500]}...")
        return

    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from trl import SFTTrainer

    model_name: str = cfg["model"]["name"]
    max_seq_len: int = cfg["model"]["max_seq_length"]

    console.print(f"\n[cyan]Loading tokenizer:[/cyan] {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = Dataset.from_dict({"text": chunk_list})
    console.print(f"[bold]Prepared {len(dataset)} training sample(s)[/bold]")

    console.print(f"\n[cyan]Loading model:[/cyan] {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")

    lora_cfg = cfg["lora"]
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    t = cfg["training"]
    training_args = TrainingArguments(
        output_dir=t["output_dir"],
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        warmup_ratio=t["warmup_ratio"],
        lr_scheduler_type=t["lr_scheduler_type"],
        logging_steps=t["logging_steps"],
        save_steps=t["save_steps"],
        fp16=t["fp16"],
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_len,
        args=training_args,
    )

    console.print("\n[bold green]Starting LoRA training...[/bold green]")
    trainer.train()

    output_dir = Path(t["output_dir"])
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    console.print(f"\n[bold green]Done! Adapter saved to {output_dir}[/bold green]")


def read_chunks_jsonl(chunks_path: Path) -> list[str]:
    """Read chunk text from a JSONL file produced by the orchestrator."""

    import json

    chunks: list[str] = []
    with open(chunks_path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get("text")
            if not isinstance(text, str):
                raise ValueError(f"Missing text field in {chunks_path}:{line_number}")
            chunks.append(text)
    return chunks


@app.command()
def train(
    chunks_path: Annotated[Path, typer.Argument(help="Path to chunked JSONL data")],
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="Path to TOML LoRA config"),
    ] = Path("configs/shakespeare_lora.toml"),
    dry_run: Annotated[bool, typer.Option(help="Load chunks only; skip training")] = False,
) -> None:
    """Train a LoRA adapter from an existing chunked JSONL file."""

    cfg = load_config(config)
    chunks = read_chunks_jsonl(chunks_path)
    train_lora(chunks, cfg, dry_run=dry_run)


if __name__ == "__main__":
    app()
