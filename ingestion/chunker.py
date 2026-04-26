"""Item-aware semantic chunker.

Strategy:
  1. Respect Item boundaries: never let a chunk cross sections.
  2. Within a section, split on paragraph boundaries first.
  3. Pack paragraphs into chunks up to `max_tokens`. If a paragraph alone
     exceeds the budget, hard-wrap it on sentence boundaries, then on token
     windows as a last resort.
  4. Add an `overlap_tokens` tail from the previous chunk for context bleed,
     unless it would cross a section boundary (we don't bleed across Items).

Token counting uses a cheap whitespace approximation (≈ words). 10-K vocabulary
is heavily English with mid-length tokens; whitespace count tracks tiktoken
within ±15% on real filings, which is enough budget signal for chunking.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from core.types import Chunk, FilingDoc, ItemKey, Section

_PARAGRAPH_SPLIT_RE = re.compile(r"\n{2,}")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


@dataclass(frozen=True)
class ChunkerConfig:
    max_tokens: int = 350
    min_tokens: int = 80
    overlap_tokens: int = 60

    def __post_init__(self) -> None:
        if self.overlap_tokens >= self.max_tokens:
            raise ValueError("overlap_tokens must be smaller than max_tokens")
        if self.min_tokens >= self.max_tokens:
            raise ValueError("min_tokens must be smaller than max_tokens")


def _tokens(s: str) -> list[str]:
    return s.split()


def _take_tail(text: str, n_tokens: int) -> str:
    toks = _tokens(text)
    return " ".join(toks[-n_tokens:]) if n_tokens > 0 else ""


def _split_paragraphs(section_text: str, base_offset: int) -> list[tuple[str, int, int]]:
    """Return (paragraph, char_start, char_end) tuples within the original text."""
    out: list[tuple[str, int, int]] = []
    cursor = 0
    for piece in _PARAGRAPH_SPLIT_RE.split(section_text):
        if not piece.strip():
            cursor += len(piece) + 2  # rough; not used downstream
            continue
        start_in_section = section_text.find(piece, cursor)
        if start_in_section == -1:
            start_in_section = cursor
        end_in_section = start_in_section + len(piece)
        out.append(
            (piece.strip(), base_offset + start_in_section, base_offset + end_in_section)
        )
        cursor = end_in_section
    return out


def _hard_wrap(
    paragraph: str, char_start: int, max_tokens: int
) -> list[tuple[str, int, int]]:
    """Split an oversized paragraph into max_tokens-sized pieces.

    Tries sentence boundaries first; falls back to a token window so a single
    pathological run-on never produces an unbounded chunk.
    """
    pieces: list[tuple[str, int, int]] = []
    sentences = _SENTENCE_SPLIT_RE.split(paragraph)

    buf: list[str] = []
    buf_tokens = 0
    buf_char_start = char_start
    cursor = char_start

    def flush() -> None:
        nonlocal buf, buf_tokens, buf_char_start
        if not buf:
            return
        text = " ".join(buf).strip()
        pieces.append((text, buf_char_start, buf_char_start + len(text)))
        buf = []
        buf_tokens = 0

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        st = _tokens(sent)

        # Sentence itself bigger than budget: chop on token windows.
        if len(st) > max_tokens:
            flush()
            for i in range(0, len(st), max_tokens):
                window = " ".join(st[i : i + max_tokens])
                pieces.append((window, cursor, cursor + len(window)))
                cursor += len(window) + 1
            buf_char_start = cursor
            continue

        if buf_tokens + len(st) > max_tokens:
            flush()
            buf_char_start = cursor
        buf.append(sent)
        buf_tokens += len(st)
        cursor += len(sent) + 1

    flush()
    return pieces


def _chunk_section(
    *, doc: FilingDoc, section: Section, cfg: ChunkerConfig, ordinal_start: int
) -> list[Chunk]:
    section_text = doc.text[section.start : section.end]
    paragraphs = _split_paragraphs(section_text, base_offset=section.start)
    if not paragraphs:
        return []

    chunks: list[Chunk] = []
    buf_text: list[str] = []
    buf_tokens = 0
    buf_start: int | None = None
    buf_end: int = 0
    overlap_tail = ""
    ordinal = ordinal_start

    def emit() -> None:
        nonlocal buf_text, buf_tokens, buf_start, buf_end, overlap_tail, ordinal
        if not buf_text or buf_start is None:
            return
        body = " ".join(buf_text).strip()
        text_with_overlap = (overlap_tail + " " + body).strip() if overlap_tail else body
        if len(_tokens(text_with_overlap)) < cfg.min_tokens and chunks:
            # Too small to stand alone — merge into previous chunk if same section.
            prev = chunks[-1]
            merged_text = (prev.text + " " + body).strip()
            chunks[-1] = prev.model_copy(
                update={
                    "text": merged_text,
                    "char_end": buf_end,
                    "token_count": len(_tokens(merged_text)),
                }
            )
        else:
            chunk_id = f"{doc.company}-{doc.year}-{section.item}-{ordinal:04d}"
            chunks.append(
                Chunk(
                    id=chunk_id,
                    text=text_with_overlap,
                    company=doc.company,
                    company_name=doc.company_name,
                    year=doc.year,
                    item=section.item,
                    section_title=section.title,
                    char_start=buf_start,
                    char_end=buf_end,
                    token_count=len(_tokens(text_with_overlap)),
                    source_url=doc.source_url,
                )
            )
            ordinal += 1
        overlap_tail = _take_tail(body, cfg.overlap_tokens)
        buf_text = []
        buf_tokens = 0
        buf_start = None

    for para_text, p_start, p_end in paragraphs:
        para_tokens = _tokens(para_text)

        # Oversized paragraph: emit buffered first, then hard-wrap this para.
        if len(para_tokens) > cfg.max_tokens:
            emit()
            for piece_text, ps, pe in _hard_wrap(para_text, p_start, cfg.max_tokens):
                buf_text = [piece_text]
                buf_tokens = len(_tokens(piece_text))
                buf_start = ps
                buf_end = pe
                emit()
            continue

        if buf_tokens + len(para_tokens) > cfg.max_tokens:
            emit()

        if buf_start is None:
            buf_start = p_start
        buf_text.append(para_text)
        buf_tokens += len(para_tokens)
        buf_end = p_end

    emit()
    return chunks


def chunk_filing(doc: FilingDoc, cfg: ChunkerConfig | None = None) -> list[Chunk]:
    """Chunk a parsed filing. Sections without text yield zero chunks."""
    cfg = cfg or ChunkerConfig()
    out: list[Chunk] = []

    sections = doc.sections or [
        Section(item=_unknown_item(), title="Full filing", start=0, end=len(doc.text)),
    ]

    for section in sections:
        ordinal_start = sum(1 for c in out if c.item == section.item)
        out.extend(
            _chunk_section(doc=doc, section=section, cfg=cfg, ordinal_start=ordinal_start)
        )
    return out


def _unknown_item() -> ItemKey:
    return "UNKNOWN"
