"""CLI entry point: finetune train --method lora ..."""

import typer
from rich.console import Console

from finetune.config import TrainingMethod

app = typer.Typer(name="finetune", help="LLM fine-tuning experiment runner")
console = Console()


@app.command()
def train(
    model: str = typer.Option(..., "--model", "-m", help="Model name or path"),
    dataset: str = typer.Option(..., "--dataset", "-d", help="Dataset name or path"),
    method: TrainingMethod = typer.Option(TrainingMethod.LORA, "--method", help="Training method"),
    output_dir: str = typer.Option("outputs", "--output-dir", "-o", help="Output directory"),
    epochs: int = typer.Option(3, "--epochs", help="Training epochs"),
    run_name: str = typer.Option("finetune-run", "--run-name", help="W&B run name"),
) -> None:
    """Launch a fine-tuning run."""
    from finetune.config import BaseTrainingConfig
    from finetune.data import load_dataset_for_training
    from finetune.models import load_model_and_tokenizer
    from finetune.trainers import run_sft
    from finetune.utils import set_seed, setup_logging
    from pathlib import Path

    setup_logging()
    config = BaseTrainingConfig(
        model_name_or_path=model,
        dataset_name=dataset,
        output_dir=Path(output_dir),
        num_train_epochs=epochs,
        run_name=run_name,
    )
    set_seed(config.seed)

    console.print(f"[bold green]Starting {method} training[/bold green]")
    console.print(f"  Model   : {model}")
    console.print(f"  Dataset : {dataset}")
    console.print(f"  Output  : {output_dir}")

    ft_model, tokenizer = load_model_and_tokenizer(model, method=method)
    train_ds = load_dataset_for_training(dataset, split=config.dataset_split, tokenizer=tokenizer)

    checkpoint = run_sft(ft_model, tokenizer, train_ds, None, config)
    console.print(f"[bold]Saved to {checkpoint}[/bold]")


@app.command()
def evaluate(
    model: str = typer.Option(..., "--model", "-m", help="Checkpoint path"),
    dataset: str = typer.Option(..., "--dataset", "-d", help="Evaluation dataset"),
    prompt: list[str] = typer.Option([], "--prompt", "-p", help="Sample prompts for generation"),
) -> None:
    """Evaluate a checkpoint: perplexity + optional generation samples."""
    from finetune.data import load_dataset_for_training
    from finetune.evaluation import compute_perplexity, generate_samples
    from finetune.models import load_model_and_tokenizer
    from finetune.utils import setup_logging

    setup_logging()
    ft_model, tokenizer = load_model_and_tokenizer(model)
    eval_ds = load_dataset_for_training(dataset, split="test")
    ppl = compute_perplexity(ft_model, tokenizer, eval_ds)
    console.print(f"Perplexity: [bold]{ppl:.2f}[/bold]")

    if prompt:
        samples = generate_samples(ft_model, tokenizer, list(prompt))
        for i, sample in enumerate(samples, 1):
            console.rule(f"Sample {i}")
            console.print(sample)
