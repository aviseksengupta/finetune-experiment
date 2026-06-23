"""Interactive continuation loop for the Shakespeare LoRA adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.prompt import Prompt

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def run(
    adapter: Annotated[
        Path,
        typer.Option("--adapter", "-a", help="Path to saved LoRA adapter directory"),
    ] = Path("outputs/llama3-shakespeare-lora"),
    base_model: Annotated[
        str,
        typer.Option("--base", "-b", help="Base model name or path"),
    ] = "meta-llama/Llama-3.2-3B",
    max_new_tokens: Annotated[int, typer.Option(help="Max tokens to generate per turn")] = 200,
    temperature: Annotated[float, typer.Option(help="Sampling temperature")] = 0.8,
    top_p: Annotated[float, typer.Option(help="Nucleus sampling probability")] = 0.95,
    keep_context: Annotated[
        bool,
        typer.Option("--context/--no-context", help="Feed previous output back as context"),
    ] = False,
) -> None:
    """Continue Shakespeare-style text from your prompt.

    Enter an opening line and the model will continue it. Type 'quit' or
    press Ctrl-C to exit. Use --context to chain generations together.
    """

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not adapter.exists():
        console.print(f"[red]Adapter not found:[/red] {adapter}")
        console.print("Train the model first with: python -m finetune_pdfs.train_shakespeare_lora")
        raise typer.Exit(1)

    console.print(f"\n[cyan]Loading base model:[/cyan] {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device != "cpu" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype, device_map=device)

    console.print(f"[cyan]Loading LoRA adapter:[/cyan] {adapter}")
    model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()

    console.print(f"\n[bold green]Model ready[/bold green] — device: {device}")
    console.print("[dim]Enter an opening line. The model will continue it in Shakespeare's style.[/dim]")
    console.print("[dim]Commands: 'quit' to exit, 'clear' to reset context.[/dim]\n")

    context = ""

    while True:
        try:
            user_input = Prompt.ask("[bold yellow]You[/bold yellow]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            console.print("[dim]Goodbye.[/dim]")
            break
        if user_input.lower() == "clear":
            context = ""
            console.print("[dim]Context cleared.[/dim]\n")
            continue

        prompt = (context + "\n\n" + user_input).lstrip() if keep_context and context else user_input

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][input_len:]
        continuation = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        console.print(f"\n[bold green]Shakespeare[/bold green]: {continuation}\n")

        if keep_context:
            context = prompt + "\n" + continuation


if __name__ == "__main__":
    app()
