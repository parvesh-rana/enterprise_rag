"""Load the held-out QA set.

Each line of `qa_set.jsonl` is one question. Authoring rules:
  - `id` is a stable string slug (used in run reports).
  - `question` is what the system gets verbatim.
  - `gold_chunk_ids` lists chunk ids that *should* appear in retrieval.
    Chunks are identified by the same `f"{company}-{year}-{item}-{ordinal:04d}"`
    scheme produced by ingestion.chunker. To preserve robustness against
    re-chunking, you may instead provide `gold_substrings` — substrings that
    must appear in at least one retrieved chunk's text. The harness scores
    a hit if EITHER set matches.
  - `filter` is optional metadata pre-filter (company / year / item).
  - `kind` is one of "factoid" | "comparative" | "multi_hop" | "unanswerable"
    (the last is the abstention probe).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

QuestionKind = Literal["factoid", "comparative", "multi_hop", "unanswerable"]


class QAFilter(BaseModel):
    company: str | None = None
    year: int | None = None
    item: str | None = None


class QAExample(BaseModel):
    id: str
    question: str
    kind: QuestionKind
    gold_chunk_ids: list[str] = Field(default_factory=list)
    gold_substrings: list[str] = Field(default_factory=list)
    filter: QAFilter = Field(default_factory=QAFilter)
    notes: str = ""

    def is_unanswerable(self) -> bool:
        return self.kind == "unanswerable"


def load_qa_set(path: Path) -> list[QAExample]:
    if not path.exists():
        raise FileNotFoundError(
            f"QA set not found at {path}. See evaluation/qa_set.jsonl."
        )
    out: list[QAExample] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.append(QAExample.model_validate_json(line))
    return out
