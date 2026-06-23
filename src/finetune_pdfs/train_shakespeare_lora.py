"""Orchestrate Shakespeare text chunking and LoRA fine-tuning."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated
from urllib.request import urlopen

import typer
from rich.console import Console

from finetune_pdfs.shakespeare_chunker import chunk_shakespeare_text, save_chunks_jsonl
from finetune_pdfs.train_lora import load_config, train_lora

app = typer.Typer(add_completion=False)
console = Console()


def load_or_download_text(source_url: str, raw_text_path: Path, *, force_download: bool) -> str:
    """Load Gutenberg text from disk, downloading it if needed."""

    if raw_text_path.exists() and not force_download:
        console.print(f"[cyan]Loading text:[/cyan] {raw_text_path}")
        return raw_text_path.read_text(encoding="utf-8")

    console.print(f"[cyan]Downloading text:[/cyan] {source_url}")
    with urlopen(source_url) as response:
        text = response.read().decode("utf-8")

    raw_text_path.parent.mkdir(parents=True, exist_ok=True)
    raw_text_path.write_text(text, encoding="utf-8")
    console.print(f"[green]Saved raw text to {raw_text_path}[/green]")
    return text


@app.command()
def run(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="Path to TOML orchestration config"),
    ] = Path("configs/shakespeare_lora.toml"),
    raw_text_path: Annotated[
        Path | None,
        typer.Option(help="Override raw Gutenberg text path from config"),
    ] = None,
    chunks_path: Annotated[
        Path | None,
        typer.Option(help="Override chunked JSONL output path from config"),
    ] = None,
    force_download: Annotated[
        bool,
        typer.Option(help="Download the Gutenberg source even if the raw file exists"),
    ] = False,
    rebuild_chunks: Annotated[
        bool,
        typer.Option(help="Rebuild chunks even if the JSONL file already exists"),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(help="Download/chunk only; skip LoRA training"),
    ] = False,
) -> None:
    """Download/load Shakespeare, chunk it semantically, then train LoRA."""

    cfg = load_config(config)
    data_cfg = cfg["data"]
    resolved_raw_text_path = raw_text_path or Path(data_cfg["raw_text_path"])
    resolved_chunks_path = chunks_path or Path(data_cfg["chunks_path"])

    model_name: str = cfg["model"]["name"]
    console.print(f"\n[cyan]Loading tokenizer for chunking:[/cyan] {model_name}")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    if resolved_chunks_path.exists() and not rebuild_chunks:
        console.print(f"[cyan]Using existing chunks:[/cyan] {resolved_chunks_path}")
        from finetune_pdfs.train_lora import read_chunks_jsonl

        chunks = read_chunks_jsonl(resolved_chunks_path)
    else:
        text = load_or_download_text(
            data_cfg["source_url"],
            resolved_raw_text_path,
            force_download=force_download,
        )
        text_chunks = chunk_shakespeare_text(
            text,
            tokenizer,
            chunk_size=data_cfg["chunk_size"],
            overlap=data_cfg["chunk_overlap"],
            min_chunk_tokens=data_cfg["min_chunk_tokens"],
        )
        save_chunks_jsonl(text_chunks, resolved_chunks_path)
        console.print(f"[green]Saved chunks to {resolved_chunks_path}[/green]")
        chunks = [chunk.text for chunk in text_chunks]

    train_lora(chunks, cfg, dry_run=dry_run)


if __name__ == "__main__":
    app()
