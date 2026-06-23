"""Semantic chunking for Project Gutenberg's Complete Works of Shakespeare."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console

console = Console()

GUTENBERG_START = "*** START OF THE PROJECT GUTENBERG EBOOK"
GUTENBERG_END = "*** END OF THE PROJECT GUTENBERG EBOOK"

ROMAN_NUMERAL = r"[IVXLCDM]+"
WORK_HEADING_RE = re.compile(
    r"^(?:THE\s+)?(?:TRAGEDY|COMEDY|HISTORY|LIFE|LIFE AND DEATH|FIRST PART|SECOND PART|THIRD PART)\b|"
    r"^(?:THE SONNETS|A LOVER'S COMPLAINT|THE PASSIONATE PILGRIM|VENUS AND ADONIS|"
    r"THE RAPE OF LUCRECE|THE PHOENIX AND THE TURTLE)\b",
    re.IGNORECASE,
)
ACT_RE = re.compile(rf"^ACT\s+{ROMAN_NUMERAL}\.?$", re.IGNORECASE)
SCENE_RE = re.compile(rf"^SCENE\s+{ROMAN_NUMERAL}\b", re.IGNORECASE)
# Sonnets in Gutenberg pg100 are numbered with Arabic numerals, centered (stripped to bare digits)
SONNET_RE = re.compile(r"^\d+\.?$")
# Speaker labels in plays: all-caps name (possibly with spaces/apostrophes) followed by a period, alone on a line
SPEAKER_RE = re.compile(r"^([A-Z][A-Z\s\'\-]+)\.\s*$")


@dataclass(frozen=True)
class TextUnit:
    """Small semantic unit that can be combined into model-sized chunks."""

    kind: str
    heading: str
    text: str


@dataclass(frozen=True)
class TextChunk:
    """A chunk ready for model training."""

    text: str
    token_count: int
    unit_count: int


def strip_gutenberg_boilerplate(text: str) -> str:
    """Remove Project Gutenberg header and license footer when markers exist."""

    start = text.find(GUTENBERG_START)
    if start != -1:
        first_newline = text.find("\n", start)
        if first_newline != -1:
            text = text[first_newline + 1 :]

    end = text.find(GUTENBERG_END)
    if end != -1:
        text = text[:end]

    return text.strip()


def normalize_text(text: str) -> str:
    """Normalize line endings while preserving Shakespeare's line structure."""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def clean_unit_text(text: str) -> str:
    """Transform play-specific markup into clean training text.

    - Speaker labels (e.g. ``HAMLET.``) become ``Hamlet:`` (title-case, no period).
    - ACT / SCENE header lines are stripped (should not appear after semantic
      splitting, but kept as a safety net for edge cases).
    """

    lines_out: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()

        # Drop residual structural headers
        if ACT_RE.match(stripped) or SCENE_RE.match(stripped):
            continue

        # Transform all-caps speaker label into readable dialogue prefix
        m = SPEAKER_RE.match(stripped)
        if m:
            name = m.group(1).strip().title()
            lines_out.append(f"{name}:")
            continue

        lines_out.append(line.rstrip())

    return "\n".join(lines_out).strip()


def split_semantic_units(text: str) -> list[TextUnit]:
    """Split Shakespeare text on work, act, scene, and sonnet boundaries."""

    units: list[TextUnit] = []
    current_lines: list[str] = []
    current_kind = "section"
    current_heading = "Opening"
    current_work = ""

    def flush() -> None:
        nonlocal current_lines
        body = clean_unit_text("\n".join(current_lines))
        if body:
            heading = current_heading
            if current_work and current_work not in heading:
                heading = f"{current_work} / {heading}"
            units.append(TextUnit(kind=current_kind, heading=heading, text=body))
        current_lines = []

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            current_lines.append("")
            continue

        if WORK_HEADING_RE.match(stripped):
            flush()
            current_work = stripped
            current_kind = "work"
            current_heading = stripped
            current_lines = []  # heading itself is metadata, not training text
            continue

        if ACT_RE.match(stripped):
            flush()
            current_kind = "act"
            current_heading = stripped
            current_lines = []  # structural delimiter, not training text
            continue

        if SCENE_RE.match(stripped):
            flush()
            current_kind = "scene"
            current_heading = stripped
            current_lines = []  # structural delimiter, not training text
            continue

        if current_work.upper() == "THE SONNETS" and SONNET_RE.match(stripped):
            flush()
            current_kind = "sonnet"
            current_heading = f"Sonnet {stripped.rstrip('.')}"
            current_lines = []  # number is metadata; poem body follows
            continue

        current_lines.append(line.rstrip())

    flush()
    return units


def split_large_unit(
    unit: TextUnit,
    tokenizer: Any,
    *,
    chunk_size: int,
    min_chunk_tokens: int,
) -> list[TextUnit]:
    """Split unusually large semantic units on paragraph boundaries."""

    token_count = len(tokenizer.encode(unit.text, add_special_tokens=False))
    if token_count <= chunk_size:
        return [unit]

    pieces: list[TextUnit] = []
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", unit.text) if paragraph.strip()]
    current: list[str] = []

    def flush() -> None:
        nonlocal current
        if not current:
            return
        text = "\n\n".join(current).strip()
        if len(tokenizer.encode(text, add_special_tokens=False)) >= min_chunk_tokens:
            pieces.append(TextUnit(kind=unit.kind, heading=unit.heading, text=text))
        current = []

    for paragraph in paragraphs:
        paragraph_tokens = tokenizer.encode(paragraph, add_special_tokens=False)
        if len(paragraph_tokens) > chunk_size:
            flush()
            for start in range(0, len(paragraph_tokens), chunk_size):
                token_slice = paragraph_tokens[start : start + chunk_size]
                if len(token_slice) < min_chunk_tokens:
                    continue
                pieces.append(
                    TextUnit(
                        kind=unit.kind,
                        heading=unit.heading,
                        text=tokenizer.decode(token_slice),
                    )
                )
            continue

        candidate = "\n\n".join([*current, paragraph]).strip()
        candidate_tokens = len(tokenizer.encode(candidate, add_special_tokens=False))
        if current and candidate_tokens > chunk_size:
            flush()
        current.append(paragraph)

    flush()
    return pieces or [unit]


def chunk_semantic_units(
    units: Iterable[TextUnit],
    tokenizer: Any,
    *,
    chunk_size: int,
    overlap: int = 0,
    min_chunk_tokens: int = 128,
) -> list[TextChunk]:
    """Pack semantic units into chunks bounded by tokenizer-measured size."""

    if overlap >= chunk_size:
        raise ValueError("chunk overlap must be smaller than chunk size")

    expanded_units: list[TextUnit] = []
    for unit in units:
        expanded_units.extend(
            split_large_unit(
                unit,
                tokenizer,
                chunk_size=chunk_size,
                min_chunk_tokens=min_chunk_tokens,
            )
        )

    chunks: list[TextChunk] = []
    current_units: list[TextUnit] = []
    current_tokens = 0

    def unit_tokens(unit: TextUnit) -> int:
        return len(tokenizer.encode(unit.text, add_special_tokens=False))

    def flush() -> None:
        nonlocal current_units, current_tokens
        if not current_units:
            return

        text = "\n\n".join(unit.text for unit in current_units).strip()
        tokens = len(tokenizer.encode(text, add_special_tokens=False))
        if tokens >= min_chunk_tokens:
            chunks.append(TextChunk(text=text, token_count=tokens, unit_count=len(current_units)))

        if overlap <= 0:
            current_units = []
            current_tokens = 0
            return

        overlap_units: list[TextUnit] = []
        overlap_tokens = 0
        for unit in reversed(current_units):
            tokens_for_unit = unit_tokens(unit)
            if overlap_units and overlap_tokens + tokens_for_unit > overlap:
                break
            overlap_units.insert(0, unit)
            overlap_tokens += tokens_for_unit

        current_units = overlap_units
        current_tokens = overlap_tokens

    for unit in expanded_units:
        tokens = unit_tokens(unit)
        if current_units and current_tokens + tokens > chunk_size:
            flush()
            if current_units and current_tokens + tokens > chunk_size:
                current_units = []
                current_tokens = 0
        current_units.append(unit)
        current_tokens += tokens

    flush()
    console.print(
        f"Created [bold]{len(chunks)}[/bold] semantic chunk(s) "
        f"from [bold]{len(expanded_units)}[/bold] unit(s)"
    )
    return chunks


def chunk_shakespeare_text(
    text: str,
    tokenizer: Any,
    *,
    chunk_size: int,
    overlap: int = 0,
    min_chunk_tokens: int = 128,
) -> list[TextChunk]:
    """Clean and semantically chunk the Complete Works of Shakespeare."""

    cleaned = normalize_text(strip_gutenberg_boilerplate(text))
    units = split_semantic_units(cleaned)
    console.print(f"Detected [bold]{len(units)}[/bold] semantic unit(s)")
    return chunk_semantic_units(
        units,
        tokenizer,
        chunk_size=chunk_size,
        overlap=overlap,
        min_chunk_tokens=min_chunk_tokens,
    )


def save_chunks_jsonl(chunks: Iterable[TextChunk], path: Path) -> None:
    """Save chunks as JSONL so training is reproducible and inspectable."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for index, chunk in enumerate(chunks):
            json.dump(
                {
                    "id": index,
                    "text": chunk.text,
                    "token_count": chunk.token_count,
                    "unit_count": chunk.unit_count,
                },
                f,
                ensure_ascii=False,
            )
            f.write("\n")
