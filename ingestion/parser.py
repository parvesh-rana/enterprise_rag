"""Parse a 10-K HTML filing into clean text + Item-level section spans.

10-K filings are messy: nested tables, inline-XBRL spans, MS-Word ghost styles.
We strip aggressively, then locate Item headings via regex on the cleaned text
because EDGAR layouts vary too much for a structural (DOM-walking) approach to
generalize across all five companies.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Final, get_args

from selectolax.parser import HTMLParser

from core.types import FilingDoc, ItemKey, Section

# Tags to drop wholesale before extracting text.
_DROP_TAGS: Final[tuple[str, ...]] = (
    "script", "style", "noscript", "head", "meta", "link",
    "iframe", "form", "input", "button", "svg",
)

# Inline-XBRL wrappers carry no useful text once their inner text is preserved.
# We don't drop them; we just unwrap by reading their text content.

_WS_RE = re.compile(r"[ \t   ]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")

# Match "Item 1A.", "ITEM 1A.", "Item 7A —", optionally followed by a title on
# the same line. We capture the item key and the full heading line for `title`.
_ITEM_HEADING_RE = re.compile(
    r"""
    ^[ \t]*                                  # optional indent
    item[ \t]+                               # 'Item'
    (?P<key>\d{1,2}[A-C]?)                   # 1, 1A, 7A, 9B, 1C ...
    [ \t]*[.—–:\-]?[ \t]*          # separator (., :, em/en dash, hyphen)
    (?P<title>[^\n]{0,200})                  # rest of the heading line
    """,
    re.IGNORECASE | re.MULTILINE | re.VERBOSE,
)

_VALID_ITEM_KEYS: Final[frozenset[str]] = frozenset(get_args(ItemKey)) - {"UNKNOWN"}


def html_to_text(html: str) -> str:
    """Strip a 10-K HTML to clean plain text suitable for downstream parsing.

    Steps:
      1. Remove non-content tags (script/style/etc).
      2. Convert <br> and block-level closings to newlines.
      3. Extract text, normalize unicode, collapse whitespace.
    """
    tree = HTMLParser(html)

    for sel in _DROP_TAGS:
        for node in tree.css(sel):
            node.decompose()

    # selectolax already inserts spaces between block elements when we ask for
    # text with `separator="\n"`; that preserves paragraph boundaries cheaply.
    text = tree.body.text(separator="\n", strip=False) if tree.body else tree.text(separator="\n")

    # Normalize unicode (10-Ks contain NBSPs, fancy dashes, fullwidth digits).
    text = unicodedata.normalize("NFKC", text)

    # Collapse runs of horizontal whitespace, trim trailing spaces per line.
    text = _WS_RE.sub(" ", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = _BLANK_LINES_RE.sub("\n\n", text).strip()
    return text


def _normalize_item_key(raw: str) -> ItemKey:
    key = raw.upper()
    if key in _VALID_ITEM_KEYS:
        return key  # type: ignore[return-value]
    return "UNKNOWN"


def find_sections(text: str) -> list[Section]:
    """Locate Item-level section spans within `text`.

    A section runs from one Item heading to the next. The 10-K table-of-contents
    (front matter) also matches Item regex; we de-duplicate by keeping the LAST
    occurrence of each Item key, which is the actual body section in every
    real 10-K we tested.
    """
    matches: list[tuple[ItemKey, str, int]] = []
    for m in _ITEM_HEADING_RE.finditer(text):
        key = _normalize_item_key(m.group("key"))
        if key == "UNKNOWN":
            continue
        title = (f"Item {m.group('key').upper()} {m.group('title').strip()}").strip()
        matches.append((key, title, m.start()))

    if not matches:
        return []

    # Keep the last occurrence per Item key (skips ToC entries).
    last_by_key: dict[ItemKey, tuple[str, int]] = {}
    order: list[ItemKey] = []
    for key, title, start in matches:
        if key not in last_by_key:
            order.append(key)
        else:
            order.remove(key)
            order.append(key)
        last_by_key[key] = (title, start)

    # Build sections by sorting on start offset, then computing end as the next
    # section's start (or end-of-text for the final one).
    ordered = sorted(
        ((k, *last_by_key[k]) for k in last_by_key),
        key=lambda t: t[2],
    )
    sections: list[Section] = []
    for i, (key, title, start) in enumerate(ordered):
        end = ordered[i + 1][2] if i + 1 < len(ordered) else len(text)
        sections.append(Section(item=key, title=title, start=start, end=end))
    return sections


def parse_filing(
    *,
    html: str,
    company: str,
    company_name: str,
    year: int,
    cik: str,
    accession: str,
    source_url: str,
) -> FilingDoc:
    """End-to-end: HTML → FilingDoc with text and Item sections."""
    text = html_to_text(html)
    sections = find_sections(text)
    return FilingDoc(
        company=company,
        company_name=company_name,
        year=year,
        cik=cik,
        accession=accession,
        source_url=source_url,
        text=text,
        sections=sections,
    )
